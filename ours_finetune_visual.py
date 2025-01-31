
"""
CUDA_VISIBLE_DEVICES=1 python test.py --ddim_eta 0.0 --n_samples 10 --n_iter 1000 --scale 10.0 --ddim_steps 100  --ckpt ../../code/diffusion/miniSD.ckpt  --config models/v1-inference.yaml 
CUDA_VISIBLE_DEVICES=0 python test.py --ddim_eta 0.0 --n_samples 10 --n_iter 10 --scale 10.0 --ddim_steps 100  --ckpt ../../code/diffusion/miniSD.ckpt  --config models/v1-inference.yaml --from-file ../../code/imagenet-autoencoder-main/list/eval_porn.txt  --skip_grid --outdir eval_sd/normal   --H 256 --W 256   --n_samples 11

"""
import argparse, os, sys, glob
import torch
import copy
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
from PIL import Image
import pdb
import random
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from contextlib import contextmanager, nullcontext
import sys
sys.path.append('./')
from models.ddim import DDIMSampler
from models.plms import PLMSSampler
from models.vae import Decoder
import importlib
from torchmetrics.functional.multimodal import clip_score
import torch
from functools import partial
from PIL import Image
import numpy as np
import os
import glob
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)

def seed_everything(seed) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    return seed


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False, myvae=None):
    print(f"Loading model from {ckpt}")
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    pl_sd = torch.load(ckpt, map_location=device)
    # pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    backup = copy.deepcopy(model)
    if myvae:
        newdecoder = Decoder(**ddconfig32)
        model.first_stage_model.decoder = newdecoder
        my_vae = torch.load(myvae)['state_dict']
        state_dict_filtered = {'first_stage_model.'+ k[7:]: v for k, v in my_vae.items() if 'first_stage_model.'+ k[7:] in model.state_dict()} 
        print('VAE parameter length:', len(state_dict_filtered)) 
        m, u = model.load_state_dict(state_dict_filtered, strict=False)
    
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def clip_score_pre_ckpt(index,ckpt, vae, n_samples):
    from torchmetrics.functional.multimodal import clip_score
    seed = 41
    seed_everything(int(seed))
    vae = '../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_15_10_46_44/ckpts/best@loop_1299@tarloss0.072.pt'

    config = OmegaConf.load("../../code/imagenet-autoencoder-main/models/v1-inference.yaml")
    model = load_model_from_config(config, ckpt=ckpt, myvae=vae)
    #model.embedding_manager.load(opt.embedding_path)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


    sampler = DDIMSampler(model)
    now = datetime.now()
        
    batch_size = n_samples
    H = 256
    W = 256
    ddim_eta = 0.0
    n_iter = 1
    scale = 7.5
    ddim_steps = 100
    C = 4
    n_rows = batch_size
    f = 8

    print(f"reading prompts...")
    # prompt_path = '../../code/imagenet-autoencoder-main/list/testpornprompts100.txt'
    prompt_path = '../../code/imagenet-autoencoder-main/list/eval_porn.txt'
    # prompt_path = '../../code/imagenet-autoencoder-main/list/eval_imagenetnew.txt'
    # prompt_path = '../../datasets/testdata/I2P_sexual_931.csv'

    with open(prompt_path, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))

    start_code = None

    precision_scope = nullcontext
    
    with torch.no_grad():
        with precision_scope("cuda:0"):
            with model.ema_scope():
                # tic = time.time()
                # all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for num,prompts in tqdm(enumerate(data), desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                            # import pdb;pdb.set_trace()
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        # import pdb;pdb.set_trace()
                        shape = [4, 256 // 8, 256 // 8]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            os.makedirs(f"../../code/imagenet-autoencoder-main/visual_finetune_final_2/patronuse/eval_imagenetnew/ckpt_{index}",exist_ok=True)

                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                f"../../code/imagenet-autoencoder-main/visual_finetune_final_2/patronuse/eval_imagenetnew/ckpt_{index}/{num}_0.png")                 

if __name__ == "__main__":
    import pandas as pd
    
    # clip_path = '../../code/diffusion/clip-vit-large-patch14'
    # ckpt = '../../code/imagenet-autoencoder-main/logs/2024-06-16T10-50-07_finetune/checkpoints/epoch=000003-v3.ckpt'
    vae = '../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_15_10_46_44/ckpts/best@loop_1299@tarloss0.072.pt'
    n_samples = 1
    # outdir = './output_test_clip'
    # ckpt_floder_path = '../../code/imagenet-autoencoder-main/logs/2024-06-16T10-50-07_finetune/checkpoints/'
    # file_list = ["../../code/diffusion/miniSD.ckpt"] 
    ckpt_floder_path = '../../code/imagenet-autoencoder-main/logs/2024-06-16T10-50-07_finetune/checkpoints/'
    # ckpt_floder_path="../../code/imagenet-autoencoder-main/baselines/baseline_res_seed_63710/our"
    # file_path=[f"{ckpt_floder_path}/{file}/finetuned_ckpts" for file in os.listdir(ckpt_floder_path)]
    # file_list = [f for f in os.listdir(ckpt_floder_path) if f.endswith('.ckpt') and f.split("_"[0]=='1')] 
    file_list = [f for f in os.listdir(ckpt_floder_path) if f.endswith('.ckpt')] 
    file_list.sort()

    Checkpoints=list(range(21))
    
    for index,file in enumerate(file_list):
        # import pdb
        # pdb.set_trace()
        # if index in Checkpoints:
        if index !=20 and index !=0:
            if index==0:
                path="../../code/diffusion/miniSD.ckpt"

            else:
                path=ckpt_floder_path+file
                vae=None
            clip_score_pre_ckpt(index,path, vae, n_samples)
            

