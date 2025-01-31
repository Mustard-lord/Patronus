"""
"""
import torch
import torch
import sys
sys.path.append('../')
from models.vae import AutoencoderKL

def init_from_ckpt(model, path, ignore_keys=list()):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    model.load_state_dict(sd, strict=False)
    print(f"Restored from {path}")
    return model

ddconfig32 = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0
    }

ddconfig64 = {
"double_z": True,
"z_channels": 3,
"resolution": 256,
"in_channels": 3,
"out_ch": 3,
"ch": 128,
"ch_mult": [1, 2, 4],
"num_res_blocks": 2,
"attn_resolutions": [],
"dropout": 0.0
}
embed_dim = 4 
model = AutoencoderKL(ddconfig=ddconfig32, embed_dim=embed_dim).cuda()
model = init_from_ckpt(model, '../../code/imagenet-autoencoder-main/assets/model.ckpt')
model = torch.nn.DataParallel(model)
torch.save({'state_dict':model.state_dict()},'../assets/pretrained.pt')