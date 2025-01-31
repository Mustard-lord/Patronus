import sys
sys.path.append('../')
import os
import argparse
import pdb
from tqdm import tqdm
import pandas as pd
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # parser.add_argument('-bs', '--batch-size', default=20, type=int)
    parser.add_argument('--batch-size', default=20, type=int)
    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--root', default='./baseline_res', type=str)                      
    parser.add_argument('--target', default='porn', type=str, choices=['celeba', 'steppenwolf', '99', 'porn'])
    parser.add_argument('--original_train_list', default="../list/imagenet_train_list.txt", type=str)
    parser.add_argument('--original_test_list', default="../list/imagenet_test_list.txt", type=str)
    parser.add_argument('--target_test_list', default="../list/porn_test_list.txt", type=str)
    parser.add_argument('--target_defense_list', default="../list/porn_defense2_list.txt", type=str)
    parser.add_argument('--target_finetune_list', default="../list/porn_finetune_list.txt", type=str)
    parser.add_argument('--truly_finetune_freq', default=1000, type=int)
    parser.add_argument('--truly_finetune_epochs', default=1, type=int)
    parser.add_argument('--gpus', default=None, type=str)
    parser.add_argument('--only_decoder', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--arch', default='32x4', type=str)
    parser.add_argument('--baseline', default='pretrained', type=str, choices=['our', 'pretrained', 'scratch'])
    parser.add_argument('--syn', default=0, type=int)
    parser.add_argument('--loss', default=None)
    parser.add_argument('--commit', default=None, type=str)
    parser.add_argument('--resume_ckpt', default=None, type=str)
    parser.add_argument('--finetune_number', default=500, type=int)
    parser.add_argument('--syn_original_test_list', default=None, type=str)
    parser.add_argument('--syn_target_defense_list', default=None, type=str)
    parser.add_argument('--syn_target_finetune_list', default=None, type=str)
    parser.add_argument('--syn_target_test_list', default=None, type=str)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adade', 'rms', 'nes'])
    parser.add_argument('--resolution', default=256, type=int)
    args = parser.parse_args()
    return args

args = get_args()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
    
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import copy
from datetime import datetime
import wandb
import torch
import os
import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
import socket
from utils import set_seed, save_args_to_file, draw, get_dataset, get_features
from torch.utils.data import TensorDataset, DataLoader

def test(model, original_testloader, device):
    test_loss = 0
    total = 0
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(original_testloader):
            # import pdb;pdb.set_trace()
            inputs = inputs.to(device)
            outputs = model(inputs)[0]
            loss = criterion(outputs, inputs) 
            test_loss += loss.item() * outputs.shape[0]
            if batch_idx == 0:
                recons = outputs
            total += inputs.size(0)
    return test_loss*1.0/total, recons

def test_finetune(save_dir, model, tar_finetuneloader, tar_testloader, epochs, lr, optimizer_select, baseline, finetune_number):
    # model = nn.DataParallel(model)
    if optimizer_select == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_select == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_select == 'adade':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-6)
    elif optimizer_select == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.0)
    elif optimizer_select == 'nes':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # sgd_optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    # nesterov_optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # adam_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # adagrad_optimizer = optim.Adagrad(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # adadelta_optimizer = optim.Adadelta(model.parameters(), lr=1e-2, rho=0.9, eps=1e-6)
    # rms_optimizer = optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0.0)
    
    criterion = nn.MSELoss()
    test_loss = -1
    wandb.define_metric("truly_finetune_epoch")
    wandb.define_metric("Truly Finetune/performance test loss", step_metric='truly_finetune_epoch')
    res = []
    for ep in tqdm(range(epochs)):
        model.train()
        with tqdm(tar_finetuneloader) as tqdm_iter:
         for inputs in tqdm_iter:
            inputs = inputs.cuda()
            outputs = model(inputs)[0]
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_iter.set_description(f'train loss {round(loss.item(),4)}')
            # import pdb;pdb.set_trace()
            test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
            res.append(test_loss)
            loss_df = pd.DataFrame()
            loss_df['mse_loss'] = res
            os.makedirs(f'./epochs_{finetune_number}_seed_{args.seed}', exist_ok=True)
            loss_df.to_csv(f'./epochs_{finetune_number}_seed_{args.seed}/mse_loss_{baseline}_{optimizer_select}_batch_size_{inputs.shape[0]}_lr_{lr}_finetune_size_{finetune_number}_epochs_{epochs}.csv', index=False)
            

        # test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
        print('finetuned test loss:', round(test_loss,5))
        wandb.log({'truly_finetune_epoch': ep})
        wandb.log({"Truly Finetune/performance test loss":test_loss})
        torch.save({'state_dict':model.state_dict()}, save_dir+f'/{ep}_{round(test_loss,4)}.pt')
    # test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
    return round(test_loss,7), recons, model

  
def main(args):
    # names = args.name if args.name is not None else f'maml{args.maml}_alpha{args.alpha}_beta{args.beta}'
    wandb.init(
    project = "new baselines",  
    entity = "pangpang",
    config = args,
    name = args.baseline,
    notes = args.notes,         
)   
    set_seed(args.seed)
    save_path = args.root +f"_{args.truly_finetune_epochs}_seed_{args.seed}" f'/{args.baseline}/'
    now = datetime.now()
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    os.makedirs(save_path, exist_ok=True)
    finetuned_save_path = save_path+'/finetuned_ckpts/'
    os.makedirs(finetuned_save_path, exist_ok=True)
    save_args_to_file(args, save_path+"args.json")
            
    wandb.log({'save path': save_path})
    wandb.config.update(args)
    
    log_dir = wandb.run.dir
    txt_file_path = save_path+"/log_dir.txt"
    with open(txt_file_path, "w") as f:
        f.write('wandb sync '+ log_dir[:-6] + '\n')
    if args.baseline == 'our' and args.resume_ckpt is not None:
        with open(txt_file_path, "w") as f:
            f.write('test ckpt: '+ args.resume_ckpt)
    from models.vae import AutoencoderKL, ddconfig32, ddconfig64, Decoder
    ddconfig = ddconfig32 if args.arch == '32x4' else ddconfig64
    embed_dim = 4 if args.arch == '32x4' else 3
    ddconfig['resolution'] = args.resolution
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=embed_dim)
    model = model.cuda()
    init_model = copy.deepcopy(model)
    model = torch.nn.DataParallel(model)

    print('=> loading ckpt ...')
    # ckpt = torch.load('../../code/imagenet-autoencoder-main/results_m/imagenet_porn/5_13_10_8_26/ckpts/best@loop_1249@tarloss0.095.pt')
    
    if args.baseline == 'scratch':
        print('==============TEST train from scratch===============')
        ckpt = torch.load('../assets/pretrained32x32x4.pt')
        model.load_state_dict(ckpt['state_dict'])  
        model.module.decoder = copy.deepcopy(init_model.decoder)
        # torch.save({'state_dict':model.state_dict()}, '../../code/imagenet-autoencoder-main/assets/pretrained32x32x4_init.pt')    
        # import pdb;pdb.set_trace()    
    elif args.baseline == 'pretrained':
        print('==============TEST normal pretraining===============')
        ckpt = torch.load('../assets/pretrained32x32x4.pt')
        model.load_state_dict(ckpt['state_dict'])   
    elif args.baseline == 'our':
        print('==============TEST our model===============')
        if args.resume_ckpt is None:
            assert(0)
        else:
            ckpt = torch.load(args.resume_ckpt)
            model.load_state_dict(ckpt['state_dict'])   

    
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    print('=> building the dataloader ...')
    target_loader, target_finetuneloader, target_testloader, original_loader, original_testloader, synthesized = get_dataset(args)
    # import pdb;pdb.set_trace()
    print('=> building the criterion ...')
    mse_criterion = nn.MSELoss()
    model.train()
    print('=> starting training engine ...')


    print('Freeze Encoder......')
    for name, module in model.named_parameters():
            module.requires_grad = False
    for name, module in model.named_parameters():
        if 'decoder' in name:
            module.requires_grad = True
        if 'post_quant_conv' in name:
            module.requires_grad = True
            
    define_wandb()
    print('       ----test finetune----')
    finetunetest_loss, outputs, testmodel = test_finetune(finetuned_save_path, copy.deepcopy(model), target_finetuneloader, target_testloader, args.truly_finetune_epochs, args.finetune_lr, args.optimizer, args.baseline, args.finetune_number)
    # draw(-1, finetunetest_loss, outputs, save_path + f'/rec/pretest/target', args.resolution)
    print(f'finetune performance: test loss is {finetunetest_loss}')  
    wandb.log({"Finetune performance test loss":finetunetest_loss})
    # print('       ----test original----')
    # original_testloss, outputs = test(copy.deepcopy(model.module), original_testloader, torch.device('cuda'))
    # draw(-1, original_testloss, outputs, save_path + f'/rec/pretest/original', args.resolution)
    # print(f'original performance: test loss is{original_testloss}')  
    # wandb.log({"Original performance test loss":original_testloss})
    

def define_wandb():
    wandb.define_metric("epoch")
    wandb.define_metric("test_index")
    wandb.define_metric("patronus train loss", step_metric='epoch')
    wandb.define_metric("Original train loss", step_metric='epoch')
    wandb.define_metric("Dos feature loss", step_metric='epoch')
    wandb.define_metric("Normal feature loss", step_metric='epoch')
    wandb.define_metric("TestTime/Finetune performance test loss", step_metric='test_index')
    wandb.define_metric("TestTime/Truly Finetune performance test loss", step_metric='test_index')
    wandb.define_metric("TestTime/Original performance test loss", step_metric='test_index')

if __name__ == '__main__':
    main(args)


