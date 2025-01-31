import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
sys.path.append('../')
# from diffusion_model.common import Resize, RandomHorizontalFlip, Normalize, ClassificationCollater
# from diffusion_model.metrics.inception import InceptionV3
# from diffusion_model.models.diffusion_unet import DiffusionUNet
# from diffusion_model.losses import MSELoss
# from diffusion_model.common import Opencv2PIL, TorchRandomHorizontalFlip, TorchMeanStdNormalize, ClassificationCollater
import torch
from torch.utils.data import random_split, Dataset
import torchvision.transforms as transforms
# from diffusion_model.diffusion_methods import DDIMSampler
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple
from ldm.models.diffusion.finetune_ddpm import LatentDiffusion
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    def __init__(self, *tensors):
        assert (all(tensors[0].size(0) == tensor.size(0) for tensor in tensors))
        self.tensors = tensors
    
    def __getitem__(self, index):
        return {'image': self.tensors[0][index], 'label': torch.tensor(0)}

    def __len__(self):
        return self.tensors[0].size(0)


def extract(v, t, x_shape):
    '''
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    '''
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def compute_beta_schedule(beta_schedule_mode, t, linear_beta_1, linear_beta_t,
                          cosine_s):
    assert beta_schedule_mode in [
        'linear',
        'cosine',
        'quad',
        'sqrt_linear',
        'const',
        'jsd',
        'sigmoid',
    ]

    if beta_schedule_mode == 'linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
    elif beta_schedule_mode == 'cosine':
        x = torch.arange(t + 1, requires_grad=False, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((x / t) + cosine_s) / (1 + cosine_s) * math.pi * 0.5)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule_mode == 'quad':
        betas = (torch.linspace(linear_beta_1**0.5,
                                linear_beta_t**0.5,
                                t,
                                requires_grad=False,
                                dtype=torch.float64)**2)
    elif beta_schedule_mode == 'sqrt_linear':
        betas = torch.linspace(linear_beta_1,
                               linear_beta_t,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)**0.5
    elif beta_schedule_mode == 'const':
        betas = linear_beta_t * torch.ones(
            t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            t, 1, t, requires_grad=False, dtype=torch.float64)
    elif beta_schedule_mode == 'sigmoid':
        betas = torch.linspace(-6,
                               6,
                               t,
                               requires_grad=False,
                               dtype=torch.float64)
        betas = torch.sigmoid(betas) * (linear_beta_t -
                                        linear_beta_1) + linear_beta_1

    return betas

class DDPMTrainer(nn.Module):

    def __init__(self,
                 beta_schedule_mode='linear',
                 linear_beta_1=1e-4,
                 linear_beta_t=0.02,
                 cosine_s=0.008,
                 t=1000,
                 train=True):
        super(DDPMTrainer, self).__init__()
        assert beta_schedule_mode in [
            'linear',
            'cosine',
            'quad',
            'sqrt_linear',
            'const',
            'jsd',
            'sigmoid',
        ]

        self.t = t

        self.beta_schedule_mode = beta_schedule_mode
        self.linear_beta_1 = linear_beta_1
        self.linear_beta_t = linear_beta_t
        self.cosine_s = cosine_s

        self.betas = compute_beta_schedule(self.beta_schedule_mode, self.t,
                                           self.linear_beta_1,
                                           self.linear_beta_t, self.cosine_s)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. -
                                                        self.alphas_cumprod)
        self.train = train

    # forward diffusion (using the nice property): q(x_t | x_0)
    def add_noise(self, x_start, t, noise):
        # import pdb;pdb.set_trace()  
        # find sqrt_alphas_cumprod according to the time step
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t,
                                        x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy

    def forward(self,sd,batch,first_stage_key,cond_key,shots,sample):
        if  isinstance(sd, torch.nn.DataParallel):

            loss,loss_dict=sd(batch,k=first_stage_key,cond_key=cond_key,bs=shots,sample=sample)
        else:
            loss,loss_dict=sd(batch,k=first_stage_key,cond_key=cond_key,bs=shots,sample=sample)
        # device = sd.device
        # text_condition = sd.cond_stage_model(text).to(device)
        # x_start = sd.first_stage_model.encode(x_img).sample()

        # loss,loss_dict = sd(x_start, text_condition, self.train)

        return loss,loss_dict

# class config:
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]
#     num_classes = None
#     input_image_size = 32
#     time_step = 1000
#     use_input_images = False
#     # if use_gradient_checkpoint, use_compile set to False
#     use_gradient_checkpoint = False
#     save_image_dir = './images'
#     model = DiffusionUNet(inplanes=3,
#                           planes=128,
#                           planes_multi=[1, 2, 2, 2],
#                           time_embedding_ratio=4,
#                           block_nums=2,
#                           dropout_prob=0.1,
#                           num_groups=32,
#                           use_attention_planes_multi_idx=[1],
#                           num_classes=num_classes,
#                           use_gradient_checkpoint=use_gradient_checkpoint)

#     trainer = DDPMTrainer(beta_schedule_mode='linear',
#                           linear_beta_1=1e-4,
#                           linear_beta_t=0.02,
#                           cosine_s=0.008,
#                           t=1000)

#     sampler = DDIMSampler(beta_schedule_mode='linear',
#                           linear_beta_1=1e-4,
#                           linear_beta_t=0.02,
#                           cosine_s=0.008,
#                           ddpm_t=1000,
#                           ddim_t=50,
#                           ddim_eta=0.0,
#                           ddim_discr_method='uniform',
#                           clip_denoised=True)
#     # load pretrained model or not
#     trained_model_path = ''

#     train_criterion = MSELoss()

#     seed = 0
#     # batch_size is total size
#     batch_size = 1500
#     # num_workers is total workers
#     num_workers = 30
#     accumulation_steps = 1
#     optimizer = (
#         'SGD',
#         {
#             'lr': 4e-6,
#             'global_weight_decay': False,
#             # if global_weight_decay = False
#             # all bias, bn and other 1d params weight set to 0 weight decay
#             'weight_decay': 1e-5,
#             'no_weight_decay_layer_name_list': [],
#             'momentum': 0,
#         },
#     )
    

#     scheduler = (
#         'CosineLR',
#         {
#             'warm_up_epochs': 0,
#             'min_lr': 1e-6,
#         },
#     )


