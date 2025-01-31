"""
"""
import os
import argparse
import pdb
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--data_list', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--arch', default='32x4', type=str)
    parser.add_argument('--image_path', default=None, type=str)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--resume_ckpt', default='./assets/pretrained32x32x4.pt', type=str)
    args = parser.parse_args()
    return args

args = get_args()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
    
import torch
import os
import warnings
warnings.filterwarnings("ignore")
from utils import load_dataset
from models.vae import AutoencoderKL, ddconfig32, ddconfig64
ddconfig = ddconfig32 if args.arch == '32x4' else ddconfig64
embed_dim = 4 if args.arch == '32x4' else 3
model = AutoencoderKL(ddconfig=ddconfig, embed_dim=embed_dim).cuda()
model = torch.nn.DataParallel(model)
if args.resume_ckpt:
    ckpt = torch.load(args.resume_ckpt)
    model.load_state_dict(ckpt['state_dict'])
listfile = ["../../code/t2igogo/list/porn_list.txt"]
imageloader = load_dataset(args, 20, listfile)
total_features = []
for images, targets in imageloader:
    # import pdb;pdb.set_trace()
    images = images.cuda()
    with torch.no_grad():
        features = model.module.encode(images)[1]
        mean , _ = torch.chunk(features, 2, dim=1)
        mean = mean.cpu()
        print(mean.shape)
        total_features += mean
torch.save(total_features, 'encode_porn_feature.pt')