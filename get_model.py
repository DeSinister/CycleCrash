
from vidnext.vidnext import Model as VidNeXt
from vidnext.convnextplusvanilla import Model as ConvNeXtVanillaTransformer
from vidnext.rensetplusNST import Model as ResNetNSTransformer
from timesformer.models.vit import TimeSformer
from vivit_pytorch.video_transformer import ViViT
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(42)
np.random.seed(42)


    
def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:]
			state_dict[new_key] = state_dict.pop(old_key)
		else:
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)
     

def init_from_pretrain_(module, pretrained, init_module):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrained)
    else:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    if init_module == 'transformer':
        replace_state_dict(state_dict)
    elif init_module == 'cls_head':
        replace_state_dict(state_dict)
    else:
        raise TypeError(f'pretrained weights do not include the {init_module} module')
    msg = module.load_state_dict(state_dict, strict=False)
    return msg

class VideoEncoder(nn.Module):
    def __init__(self, model_type='VidNeXt', task_type='binclass', num_classes=5, segment_length=1):
        super().__init__() 
        embed_dims = None
        if model_type == 'VidNeXt':
            self.vid_trans = VidNeXt(segment_length=round(segment_length*30))
            embed_dims=1024
        if model_type == 'ConvNeXtVanillaTransformer':
            self.vid_trans = ConvNeXtVanillaTransformer(segment_length=round(segment_length*30))
            embed_dims=1024
        if model_type == 'ResNetNSTtransformer':
            self.vid_trans = ResNetNSTransformer(segment_length=round(segment_length*30))
            embed_dims = 512
        if model_type == 'TimeSformer':
            self.vid_trans = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='timesformer\\TimeSformer_divST_8x32_224_K400.pyth')
            self.vid_trans.model.head = nn.Identity()
            embed_dims = 768
        if model_type == 'ViViT':
            self.vid_trans = ViViT(num_frames=16,
                  img_size=224,
                  patch_size=16,
                  embed_dims=768,
                  in_channels=3,
                  attention_type='fact_encoder',
                  return_cls_token=True)
            init_from_pretrain_(self.vid_trans, "vivit_pytorch\\vivit_model.pth", init_module='transformer')
            embed_dims=768
        if model_type in ['slow_r50', 'r2plus1d_r50', 'x3d_xs', 'x3d_s', 'x3d_m']:
            self.vid_trans = torch.hub.load('facebookresearch/pytorchvideo', model_type, pretrained=True)
            self.vid_trans.blocks[-1].proj = nn.Identity()
            embed_dims = 2048

        self.bn1 = nn.BatchNorm1d(embed_dims, eps=1e-5)
        if task_type == 'binclass' or task_type == 'reg':
            self.fc1 = nn.Linear(embed_dims, 1)
        elif task_type == 'multiclass':
            self.fc1 = nn.Linear(embed_dims, num_classes)
        self.drop = nn.Dropout(p=0.25)


    def forward(self, x):
        x = self.vid_trans(x)       
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.drop(x)
        return x
