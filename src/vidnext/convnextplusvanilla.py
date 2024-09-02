import torch
import torch.nn as nn
import torch.optim as optim
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding
from .spatial_models.convnext import convnext_base
from .utils.arg_parser import parse_yaml_file


class Model(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, segment_length=30):
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
        activation = config.get('activation', {})
        self.enc_embedding = DataEmbedding(enc_in, d_model, enc_in, 15,
                                           dropout_cfg)
        self.dec_embedding = DataEmbedding(dec_in, d_model, enc_in, 15,
                                           dropout_cfg)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout_cfg,
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
                        FullAttention(True, factor, attention_dropout=dropout_cfg, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout_cfg, output_attention=False),
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

   
    def forward(self, x, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x = self.spatial_model(x)
        x = x.view(x_shape[0], x_shape[1], -1)
        x_enc = x.clone()
        x_dec = x.clone()
        x_mark_enc = torch.ones_like(x_enc)
        x_mark_dec = torch.ones_like(x_dec)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return torch.squeeze(dec_out[:, -self.pred_len:, :], dim=1), attns
        else:
            return torch.squeeze(dec_out[:, -self.pred_len:, :], dim=1)  # [B, L, D]
        


if __name__ == "__main__":
    model = Model(segment_length=30).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    bs = 2
    m = torch.rand(bs, 30, 3, 224, 224).cuda()
    pred = model(m)
    print(pred)
    loss = criterion(pred, torch.rand(bs, 1).cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()   