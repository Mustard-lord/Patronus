import argparse, os, datetime, copy
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--name', default="draft", type=str)
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--fast_lr', default=0.01, type=float)
    parser.add_argument('--alpha', default=1, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=0.5, type=float, help='coefficient of natural lr')
    parser.add_argument('--shots', default=6, type=int)
    parser.add_argument('--test_iterval', default=100, type=int)
    parser.add_argument('--root', default='./results_antift', type=str)                      
    parser.add_argument('--batches', default=6, type=int)
    parser.add_argument('--maml', default='m', type=str, choices=['s', 'm', 'a', 'x'])
    parser.add_argument('--gpus', default='2', type=str)
    parser.add_argument('--same_path', default=1, type=int)
    parser.add_argument('--mixed', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume_ckpt', default=None, type=str)
    parser.add_argument('--disalbe_wandb', default=True, type=bool)
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--alpha_LoRA', default=8, type=int)
    parser.add_argument('--unlearn', default="black noise", type=str)
    parser.add_argument('--strategy', required=True, type=str)
    parser.add_argument('--LoRAmodule', default="to_q,to_k,to_v,to_out.0", type=str)
    parser.add_argument('--adapter_name', default="LoRA_unet", type=str)
    parser.add_argument('--checkpoint', type=str,default=None)
    parser.add_argument('--FTAT_epoch', type=int,default=1)
    parser.add_argument('--module', default="whole", type=str)
    parser.add_argument('--resume', default=None, type=int)

    # ----- Add ----
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adade', 'rms', 'nes'])
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    # parser.add_argument('--detach', action="store_true")
    args = parser.parse_args()
    return args

args = get_args()
if args.gpus:
    devices_id = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices_id)
import numpy as np

from omegaconf import OmegaConf

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import pdb
from tqdm import tqdm
from src.trainer import DDPMTrainer
from peft import LoraConfig,LoraModel,get_peft_model,PeftModel

import torch
import numpy as np
import copy
from datetime import datetime
import wandb
import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
from src.patronus_losses import fast_adapt_new
from src.utils import load_model_from_config,save_args_to_json, test_finetune
from tools.draw import draw_img,draw_loss_curves,plot_FT_loss
from tools.param_ctrl import log_grad_module, unet_and_lora_require_grad,unet_and_lora_require_grad_detach
def main(args):
    # import pdb;pdb.set_trace()
    now = datetime.now()
    names = f'{args.unlearn}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    if not args.disalbe_wandb:
        wandb.init(
        project = "Robust diffusion",  
        entity = "longzai6",
        config = args,
        name = names,
        # notes = args.notes,         
    
    )   
    save_path = args.root + '/'
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_optimize-{args.optimizer}_FTlr-{args.finetune_lr}_FTepochs-{args.FTAT_epoch}_strategy-{args.strategy}_module-{args.module}/'
    if args.resume_ckpt and args.same_path:
        print(f'Resuming from {args.resume_ckpt}')
        index = args.resume_ckpt.find("ckpt")
        if index >0:
            resumed_save_path = args.resume_ckpt[:index]
            save_path = resumed_save_path
            print(f'Save to {save_path}...continue training...')
            
    else:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'/checkpoints/', exist_ok=True)
    # import pdb;pdb.set_trace()
    config = OmegaConf.load("./config/ldm_FTPokemon.yaml")
    model = load_model_from_config(config, '../../code/diffusion/miniSD.ckpt')
    
    print('=> loading ckpt ...')
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt['state_dict'])
        if 'loop' in ckpt.keys():
            if args.same_path:
                args.start_epoch = ckpt['loop']
    
    
    target_list=args.LoRAmodule.split(",")
    unet_lora_config = LoraConfig(
            r=int(args.rank),
            lora_alpha=int(args.alpha_LoRA),
            init_lora_weights="gaussian",
            target_modules=target_list,
        )
    if args.module =="lora":
        model=PeftModel.from_pretrained(model,args.checkpoint,adapter_name=args.adapter_name,is_trainable=True)
        model=model.merge_and_unload(progressbar=True)
    elif args.module=="whole":
        if args.checkpoint!= None:
            m, u = model.load_state_dict(torch.load(args.checkpoint), strict=False)
    elif args.module =="pretrained":
        print("using pretrained model\n")
    else:
        raise ValueError(f"{args.module} is not allowed")
        # import pdb;pdb.set_trace()

    if args.strategy=="scratch":
        unet_and_lora_require_grad(model,enable=True,init=True)
        print("start train from scratch")
    elif args.strategy=="FTLoRA":
        model=get_peft_model(model,unet_lora_config,adapter_name=args.adapter_name,mixed=False)
        # unet_and_lora_require_grad(model,enable=True,init=False)
    elif args.strategy=="whole":
        unet_and_lora_require_grad(model,enable=True,init=False)
    else:
        raise ValueError(f"{args.strategy} is not supported")

    model_structure = str(model)
    with open(os.path.join(save_path,"model_struc_new.txt"), 'w') as f:
        f.write(model_structure)

    model = torch.nn.DataParallel(model)
   
    print('=> building the dataloader ...')   
    data = instantiate_from_config(config.data)
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    porn_loader, benign_loader, target_finetuneloader = data.train_porn_dataloader,data.train_norm_dataloader, data.target_finetuneloader


    print(f'===> length of dataloader: target_finetuneloader:{len(target_finetuneloader)}')
    save_args_to_json(args,config,os.path.join(save_path,"args.json"),FTiteration=len(target_finetuneloader)*args.FTAT_epoch)        

            
    print('=> building the criterion ...')
    model.train()
    print('=> starting training engine ...')
  
    trainer=DDPMTrainer('linear',
                        config.model.params.linear_start,
                        config.model.params.linear_end,
                        0.008,
                        config.model.params.timesteps,
                        train=True)

    prompt_paths={
        "porn":'../../code/imagenet-autoencoder-main/list/eval_porn.txt',
        "norm":'../../code/imagenet-autoencoder-main/list/eval_pokemon.txt',
    }
    log_grad_module(model,save_path,"requires_grad_parameters")

      
    ckpt_path = ''
    # model = torch.load(ckpt_path)
    img_store_path=save_path+"/"+args.adapter_name
    finetuned_model = test_finetune(model.module, trainer, target_finetuneloader, args.finetune_lr, args.optimizer, epoch=args.FTAT_epoch,save_path=save_path,img_store_path=img_store_path,prompt_paths=prompt_paths,resume=args.resume)
    # draw_img(finetuned_model.module,img_store_path,prompt_paths)
    # try:
    #     plot_FT_loss(save_path)
    # except:
    #     print("plot_FT_loss has error")
    if args.strategy== "whole":
        torch.save(finetuned_model.state_dict(), save_path+"/finetuned_model.pth")
    elif args.strategy=="FTLoRA":
        model.module.save_pretrained(save_path)
    else:
        raise ValueError(f"{args.strategy} is not supported for save")
if __name__ == '__main__':

    main(args)


