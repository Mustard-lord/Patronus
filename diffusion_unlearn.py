import argparse, os, datetime
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--name', default="draft", type=str)
    parser.add_argument('--epochs', default=50000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--fast_lr', default=0.01, type=float)
    parser.add_argument('--alpha', default=1, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=0.01, type=float, help='coefficient of natural lr')
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--shots', default=0, type=int)
    parser.add_argument('--test_iterval', default=100, type=int)
    parser.add_argument('--root', default='./results_LoRA', type=str)                      
    parser.add_argument('--finetune_epochs', default=1, type=int)
    parser.add_argument('--batches', default=1, type=int)
    parser.add_argument('--maml', default='m', type=str, choices=['s', 'm', 'a', 'x'])
    parser.add_argument('--gpus', default='1', type=str)
    parser.add_argument('--same_path', default=0, type=int)
    parser.add_argument('--mixed', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume_ckpt', default=None, type=str)
    parser.add_argument('--disalbe_wandb', default=True, type=bool)
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--alpha_LoRA', default=8, type=int)
    parser.add_argument('--module', default="whole", type=str)
    parser.add_argument('--strategy', default="unlearn", type=str)
    parser.add_argument('--LoRAmodule', default="to_q,to_k,to_v,to_out.0", type=str)
    parser.add_argument('--adapter_name', default="LoRA_unet", type=str)
    parser.add_argument('--sample', type=str,default="random")
    parser.add_argument('--resume', default=0, type= int)
    parser.add_argument('--targets', type=str,default="diffusion_model")
    parser.add_argument('--checkpoint', type=str,default=None)

    args = parser.parse_args()
    return args

args = get_args()
    # import pdb;pdb.set_trace()
if args.gpus:
    devices_id = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices_id)
import numpy as np

from omegaconf import OmegaConf
from peft import LoraConfig,LoraModel,get_peft_model,PeftModel

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import pdb
from tqdm import tqdm
from src.trainer import DDPMTrainer
from peft import LoraConfig,LoraModel,get_peft_model


import torch
import numpy as np
import copy
from datetime import datetime
import wandb
# from tools.image_loss import vggloss
import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
# from utils import set_seed, save_args_to_file, test, test_finetune, draw, get_dataset, get_features, homemade_test_finetune, get_featureset, weighted
from src.patronus_losses import fast_adapt_unlearn
from src.utils import load_model_from_config,save_args_to_json
from tools.draw import draw_img,draw_loss_curves
from torch.optim.lr_scheduler import StepLR
from tools.param_ctrl import unet_and_lora_require_grad,log_grad_module

def main(args):
    now = datetime.now()
    names = f'{args.module}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}/'
    if not args.disalbe_wandb:
        wandb.init(
        project = "Bpalm_unlearn",  
        entity = "longzai6",
        config = args,
        name = names,
        # notes = args.notes,         
    
    )   
    save_path = args.root + '/'
    save_path = save_path + '/' + f'{args.module}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_alpha_{args.alpha}_beta_{args.beta}_lr_{args.lr}'
    if args.resume_ckpt and args.same_path:
        print(f'Resuming from {args.resume_ckpt}')
        save_path = f'{args.resume_ckpt}/../../..'
        print(f'Save to {save_path}...continue training...')
            
    else:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'/checkpoints/', exist_ok=True)

    config = OmegaConf.load("./config/ldm_unlearn.yaml")
    model = load_model_from_config(config, '../../code/diffusion/miniSD.ckpt')
    log_grad_module(model,save_path,"grad_init")

    print('=> loading ckpt ...')
    if args.module =="lora":
        if args.resume_ckpt:
            model=PeftModel.from_pretrained(model,args.resume_ckpt,is_trainable=True,adapter_name=args.adapter_name)
        else:   
            target_list=args.LoRAmodule.split(",")
            unet_lora_config = LoraConfig(
                    r=int(args.rank),
                    lora_alpha=int(args.alpha_LoRA),
                    init_lora_weights="gaussian",
                    target_modules=target_list,
                )
            model=get_peft_model(model,unet_lora_config,adapter_name=args.adapter_name,mixed=False)
    else:
        targets=args.targets.split(",")
        if args.checkpoint !=None:
            m, u = model.load_state_dict(torch.load(args.checkpoint), strict=False)
        # import pdb;pdb.set_trace()
        unet_and_lora_require_grad(model,enable=True,targets=targets)
    model = torch.nn.DataParallel(model)

    # model = model.cuda()
    model_structure = str(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    with open(os.path.join(save_path,"model_struc_new.txt"), 'w') as f:
        f.write(model_structure)

    opt1 = optim.Adam(model.parameters(filter(lambda p: p.requires_grad, model.parameters())), args.lr, weight_decay=args.wd)
    scheduler = StepLR(opt1, step_size=args.epochs//3, gamma=0.1)
   
    print('=> building the dataloader ...')   
    data = instantiate_from_config(config.data)
    # data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    porn_loader, benign_loader = data.train_porn_dataloader,data.train_norm_dataloader

    print('preparing iteration...')
    porn_iter = iter(porn_loader)
    benign_iter = iter(benign_loader)
    print(f'===> length of dataloader: target_iter:{len(porn_loader)}, original_iter:{len(benign_loader)}')
    save_args_to_json(args,config,os.path.join(save_path,"args.json"),step_size=args.epochs//3, gamma=0.1,target_iter=len(porn_loader), original_iter=len(benign_loader),total_iteration=args.epochs*args.batches)  

            
    model.train()
    best = 100
 
  
    trainer=DDPMTrainer('linear',
                        config.model.params.linear_start,
                        config.model.params.linear_end,
                        0.008,
                        config.model.params.timesteps,
                        train=True)

    tester=DDPMTrainer('linear',
                        config.model.params.linear_start,
                        config.model.params.linear_end,
                        0.008,
                        config.model.params.timesteps,
                        train=False)
    prompt_paths={
        "porn":'../../code/imagenet-autoencoder-main/list/eval_porn.txt',
        "norm":'../../code/imagenet-autoencoder-main/list/eval_imagenetnew.txt',

    }
    log_grad_module(model,save_path,"grad after control")

    for total_loop in range(0, args.epochs):
        print(f'\n================LOOP {total_loop}=================')

        pornbatches = []
        benignbatches = []

        for _ in range(args.batches):
            try:
                batch = next(porn_iter)
                pornbatches.append(batch)
            except StopIteration:
                porn_iter = iter(porn_loader)
                batch = next(porn_iter)
                pornbatches.append(batch)
        for _ in range(args.batches):
            try:
                batch = next(benign_iter)
                benignbatches.append(batch)
            except StopIteration:
                benign_iter = iter(benign_loader)
                batch = next(benign_iter)
                benignbatches.append(batch)

        if args.resume!=0:
            if total_loop<args.resume:
                continue
        opt1.zero_grad()
        
        loss = fast_adapt_unlearn(args, total_loop,pornbatches, benignbatches, model, 5, trainer, tester, save_path) 
        
        loss.backward()
        opt1.step()
        scheduler.step()
        draw_loss_curves(save_path)

        if (total_loop+1)%100==0 or total_loop==0:
            checkpoint=f"./{save_path}/checkpoints/ckpt_{total_loop}"
            img_store_path=checkpoint+"/"+args.adapter_name
            if args.module=="lora":
                draw_img(model.module,img_store_path,prompt_paths)
            else:
                draw_img(model.module,img_store_path,prompt_paths)

        if ((total_loop+1)%100==0):
            checkpoint=f"./{save_path}/checkpoints/ckpt_{total_loop}"
            if args.module=="lora":
                model.module.save_pretrained(checkpoint)
            else:
                torch.save(model.module.state_dict(), checkpoint+"/finetuned_model.pth")

if __name__ == '__main__':

    main(args)


