
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from video_transformer import ViViT


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

num_frames = 8
frame_interval = 32
num_class = 400
pretrain_pth = r"vivit_model.pth"
num_frames = num_frames * 2
model = ViViT(num_frames=num_frames,
                  img_size=224,
                  patch_size=16,
                  embed_dims=768,
                  in_channels=3,
                  attention_type='fact_encoder',
                  return_cls_token=True)

msg_trans = init_from_pretrain_(model, pretrain_pth, init_module='transformer')
print(model(torch.rand(2, 16, 3, 224, 224)).shape)