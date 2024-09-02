import torch
import torch.nn as nn
import torch.optim as optim
from .ns_layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .ns_layers.SelfAttention_Family import DSAttention, AttentionLayer
from .ns_layers.Embed import DataEmbedding
from .spatial_models.convnext import convnext_base
from .utils.arg_parser import parse_yaml_file


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class Model(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self, segment_length = 30):
        super(Model, self).__init__()
        
        self.spatial_model = convnext_base(pretrained=True)
        self.spatial_model.head = nn.Identity()
        self.seq_len = segment_length #configs.seq_len
        self.label_len = self.seq_len - 1 #configs.label_len
        self.pred_len = 1 #configs.pred_len
        self.output_attention = False #configs.output_attention

        # Embedding
        filename = '.\\vidnext\\config.yaml'
        config = parse_yaml_file(filename)
        spatial_output_dim = config.get('output_spatial_dim', {}).get('convnext')
        dropout_cfg = config.get('dropout_cfg', {})
        factor = config.get('factor', {})
        d_layers = config.get('d_layers', {})
        e_layers = config.get('e_layers', {})
        enc_in = spatial_output_dim 
        dec_in = spatial_output_dim 
        c_out = spatial_output_dim
        d_model = spatial_output_dim
        n_heads=config.get('n_heads', {})
        d_ff = config.get('d_ff', {})
        p_hidden_dims = config.get('p_hidden_dims', {})
        p_hidden_layers = config.get('p_hidden_layers', {})
        activation = config.get('activation', {})
        self.enc_embedding = DataEmbedding(enc_in, d_model, enc_in, 15,
                                           dropout_cfg)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dec_in, 15,
                                           dropout_cfg)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, factor, attention_dropout=dropout_cfg,
                                      output_attention=self.output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout_cfg,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, factor, attention_dropout=dropout_cfg, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        DSAttention(False, factor, attention_dropout=dropout_cfg, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout_cfg,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

        self.tau_learner   = Projector(enc_in=enc_in, seq_len=self.seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=enc_in, seq_len=self.seq_len, hidden_dims=p_hidden_dims, hidden_layers=p_hidden_layers, output_dim=self.seq_len)
        

    def forward(self, x, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x = self.spatial_model(x)
        x = x.view(x_shape[0], x_shape[1], -1)
        x_enc = x.clone()
        x_dec = x.clone()
        x_mark_enc = torch.ones_like(x_enc)
        x_mark_dec = torch.ones_like(x_dec)

        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

        tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar    
        delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc
        if self.output_attention:
            return torch.squeeze(dec_out[:, -self.pred_len:, :], dim=1), attns
        return torch.squeeze(dec_out[:, -self.pred_len:, :], dim=1)  # [B, L, D]

if __name__ == "__main__":
    model = Model(segment_length=30).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    bs = 18
    m = torch.rand(bs, 30, 3, 224, 224).cuda()
    pred = model(m)
    print(pred)
    loss = criterion(pred, torch.rand(bs, 1).cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()   