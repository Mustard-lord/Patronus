import matplotlib.pyplot as plt
import sys
sys.path.append('./')

from ldm.util import instantiate_from_config,load_model_from_config,chunk
from peft import LoraConfig,LoraModel,get_peft_model,PeftConfig,PeftModel
from omegaconf import OmegaConf
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from datetime import datetime
from contextlib import contextmanager, nullcontext
from tqdm import tqdm, trange
from einops import rearrange
from PIL import Image
import numpy as np
from ldm.modules.ema import LitEma
import pandas as pd
from ldm.lora_util import get_mixed_lora,get_loradict
import os
def draw_loss_curves(file_path):
    
    epochs = []
    ED_loss = []
    ND_loss = []
    cal_loss = []

    
    loss_path=file_path+"/log.txt"
    with open(loss_path, 'r') as file:
        for line in file:
            
            line = line.strip()  
            if line.startswith("epoch"):
                parts = line.split(",")
                
                
                epoch = int(parts[0].split(":")[1])
                ed_loss = float(parts[1].split(":")[1])
                nd_loss = float(parts[2].split(":")[1])
                cal_loss_value = float(parts[3].split(":")[1])

                
                epochs.append(epoch)
                ED_loss.append(ed_loss)
                ND_loss.append(nd_loss)
                cal_loss.append(cal_loss_value)

    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  

    
    axs[0].plot(epochs, ED_loss, label='Evaluation Data loss', marker='o', color='blue')
    axs[0].set_title('ED Loss Curve')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('ED Loss')
    axs[0].grid(True)
    axs[0].legend()

    
    axs[1].plot(epochs, ND_loss, label='Benign Data loss', marker='o', color='green')
    axs[1].set_title('ND Loss Curve')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('ND Loss')
    axs[1].grid(True)
    axs[1].legend()

    
    axs[2].plot(epochs, cal_loss, label='patronus loss', marker='o', color='red')
    axs[2].set_title('Cal Loss Curve')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Cal Loss')
    axs[2].grid(True)
    axs[2].legend()

    
    plt.tight_layout()

    
    plt.savefig(f'{file_path}/loss_curves_combined.png', format='png')

def draw_img(model,checkpoint,prompt_paths):
    sampler = DDIMSampler(model)
    now = datetime.now()
        
    batch_size = 1
    H = 256
    W = 256
    ddim_eta = 0.0
    n_iter = 1
    scale = 7.5
    ddim_steps = 100
    C = 4
    n_rows = batch_size
    f = 8

    for key,prompt_path in prompt_paths.items():
        if prompt_path.endswith(".txt"):
            with open(prompt_path, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
        elif prompt_path.endswith(".csv"):
            data_all = pd.read_csv(prompt_path)
            data = data_all['text'].tolist()
            data = list(chunk(data, batch_size))

        start_code = None

        precision_scope = nullcontext
        # import pdb;pdb.set_trace()

        with torch.no_grad():
            # with precision_scope("cuda:0"):
                # import pdb;pdb.set_trace()
                # with model.base_model.ema_scope():
                    # tic = time.time()
                    # all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        for num,prompts in tqdm(enumerate(data), desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [4, 256 // 8, 256 // 8]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
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

                                if key=="porn":
                                    os.makedirs(checkpoint+"/image_p",exist_ok=True)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        checkpoint+f"/image_p/{num}.png")    
                                else:
                                    os.makedirs(checkpoint+"/image_n",exist_ok=True)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        checkpoint+f"/image_n/{num}.png")  



def plot_FT_loss(file_path):
    epochs = []
    losses = []
    
    
    loss_path=file_path+"/log.txt"

    with open(loss_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("iteraton:"):
                parts = line.split(",")
                epoch = int(parts[0].split(":")[1])  
                loss = float(parts[1].split(":")[1])  
                epochs.append(epoch)
                losses.append(loss)
    
    
    epochs = np.array(epochs)
    losses = np.array(losses)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='FT Loss')
    
   
    plt.title('Fine-tuning Loss Curve')
    plt.xlabel('iteraton')
    plt.ylabel('FT Loss')
    
    
    plt.grid(True)
    plt.legend()
    
    
    plt.show()
    plt.savefig(f'{file_path}/loss_curves_combined.png', format='png')

if __name__=="__main__":
    draw_loss_curves("results_LoRA/9_11_16_1_7_black noise_alpha-1.0_beta-0.05")