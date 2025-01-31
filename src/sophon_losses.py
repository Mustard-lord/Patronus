import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import pdb
import random
# from utils import draw
import wandb
import matplotlib.pyplot as plt
from tools.param_ctrl import unet_and_lora_require_grad,log_grad_module
def fast_adapt_abandon(epoch,batch1, batch2, learner, shots, trainer,loss_strategy):

    black_tenosr_AD=torch.zeros(shots,3,256,256)
    black_tenosr_ED=torch.zeros(len(batch1[0]['image'])-shots,3,256,256)
    for AD in batch1:
        # AD_loss,AD_loss_dict = trainer(learner.module,black_tenosr_AD.cuda(),AD['txt'][:shots])
        AD_loss,AD_loss_dict = trainer(learner.module,batch1[0]['image'][:shots].cuda(),AD['txt'][:shots])# 
        if loss_strategy=="black_img":
            learner.adapt(AD_loss)
        else:
            learner.adapt(AD_loss_dict['black_loss'])

    # print("ED loss\n")
    # import pdb;pdb.set_trace()
    for ED in batch1:
        if loss_strategy=="black_img":
            ED_loss, ED_loss_dict = trainer(learner.module,black_tenosr_ED.cuda(),ED['txt'][shots:])
        #black noise strategy
        else:
            ED_loss, ED_loss_dict = trainer(learner.module,batch1[0]['image'][shots:].cuda(),ED['txt'][shots:])
    # loss1 = loss1+ED_loss
    normal_data  = batch2

    for ND in normal_data:
        ND_loss, ND_loss_dict = trainer(learner.module,ND['image'].cuda(),ND['txt'])
    # loss2 = loss2+ND_loss
    
    # import pdb;pdb.set_trace()    
    print("loss end\n")

    # ED_loss=ED_loss/(len(batch1[0]['image'])-shots)
    # ND_loss=ND_loss/len(batch2[0]['image'])
    if loss_strategy=="black_img":
        print('Query set loss', round(ED_loss.item(),2))
        print(f'Original train loss {ND_loss}') 
        # wandb.log({"epoch": epoch, "Query set loss": ED_loss.item(), "Original train loss": ND_loss,"cal_loss":0.5*ND_loss + 0.5*ED_loss})
    else:
        print('Query set loss', round(ED_loss_dict['black_loss'].item(),2))
        print(f'Original train loss {ND_loss}') 
        # wandb.log({"epoch": epoch,
        #  "Query set loss": round(ED_loss_dict['black_loss'].item(),2), 
        #  "Original train loss": ND_loss,
        #  "cal_loss":0.5*ND_loss + 0.5*ED_loss_dict['black_loss'].item()})
    if loss_strategy=="black_img":
        return 0.5*ND_loss + 0.5*ED_loss
    else:
        return 0.5*ND_loss + 0.5*ED_loss_dict['black_loss']


def fast_adapt_new(args, epoch, batch1, batch2, learner, shots, trainer, tester, save_path):
    ADloss_total=0
    EDloss_total=0
    NDloss_total=0
    if args.strategy=="unlearn":
        shots=0
    # if args.module != "whole" :
    unet_and_lora_require_grad(learner.module,enable=True)

    for AD in batch1:
        if args.strategy == "anti_finetune":
            log_grad_module(learner.module,save_path,"grad_when_adapt")

            AD_loss,AD_loss_dict = trainer(learner.module,AD,'image','txt',shots,sample="random")
            if len(args.gpus.split(","))>1:
                learner.adapt(sum(AD_loss))
            else:
                learner.adapt(AD_loss)

        # learner.module.set_adapter(args.adapter_name)

        # unet_and_lora_require_grad(learner.module,enable=False)
        # import pdb;pdb.set_trace()
        log_grad_module(learner.module,save_path,"grad_when_anti")

        ED_loss, ED_loss_dict = trainer(learner.module,AD,'image','txt',-(len(AD['image'])-shots),sample=args.sample)
        if len(args.gpus.split(","))>1:
            EDloss_total+=sum(ED_loss_dict['black_loss'])
        else:
            EDloss_total+=ED_loss_dict['black_loss']

    log_grad_module(learner.module,save_path,"grad_when_ND")

    for ND in batch2:
        ND_loss, ND_loss_dict = trainer(learner.module,ND,'image','txt',None,sample="random")
        if len(args.gpus.split(","))>1:
            NDloss_total+=sum(ND_loss)
        else:
            NDloss_total+=ND_loss

    EDloss_total /= len(batch1)
    NDloss_total /= len(batch2)
    query_set_perloss = EDloss_total.item()
    original_train_perloss = NDloss_total.item()
    cal_loss = args.alpha*NDloss_total + args.beta*EDloss_total
    # cal_loss = args.alpha*NDloss_total 
    print('Query set per loss', query_set_perloss)
    print(f'Original train per loss {original_train_perloss}')
    with open(f'{save_path}/log.txt', 'a') as file: 
        file.write(f"epoch:{epoch},ED_loss:{query_set_perloss},ND_loss: {original_train_perloss},cal_loss:{cal_loss}\n") 
    # wandb.log({"epoch": epoch,
    #  "Query set per loss": query_set_perloss, 
    #  "Original train per loss": original_train_perloss,
    #  "cal_loss":cal_loss})

    return cal_loss


# def fast_adapt_new(args, epoch, batch1, batch2, learner, shots, save_path):
#     ADloss_total=0
#     EDloss_total=0
#     NDloss_total=0
#     if args.strategy=="unlearn":
#         shots=0
#     # import pdb;pdb.set_trace()
#     unet_and_lora_require_grad(learner.module,enable=True)

#     for AD in batch1:
#         if args.strategy == "anti_finetune":
#             # import pdb;pdb.set_trace()
#             log_grad_module(learner.module,save_path,"grad_when_adapt")
#             AD_loss,loss_dict=learner.module.shared_step(batch=AD,k='image',cond_key='txt',bs=shots)
#             learner.adapt(AD_loss)
#         log_grad_module(learner.module,save_path,"grad_when_anti")
#         ED_loss,ED_loss_dict=learner.module.shared_step(batch=AD,k='image',cond_key='txt',bs=shots)

#         ED_loss, ED_loss_dict = trainer(learner.module,AD,'image','txt',-(len(AD['image'])-shots))
#         EDloss_total+=ED_loss_dict['black_loss']
    
#     log_grad_module(learner.module,save_path,"grad_when_ND")

#     for ND in batch2:
#         ND_loss, ND_loss_dict = trainer(learner.module,ND,'image','txt',None)
#         NDloss_total+=ND_loss

#     EDloss_total /= len(batch1)
#     NDloss_total /= len(batch2)
#     query_set_perloss = EDloss_total.item()
#     original_train_perloss = NDloss_total.item()
#     cal_loss = args.alpha*NDloss_total + args.beta*EDloss_total
#     # cal_loss = args.alpha*NDloss_total 
#     print('Query set per loss', query_set_perloss)
#     print(f'Original train per loss {original_train_perloss}')
#     with open(f'{save_path}/log.txt', 'a') as file: 
#         file.write(f"epoch:{epoch},ED_loss:{query_set_perloss},ND_loss: {original_train_perloss},cal_loss:{cal_loss}\n") 
#     # wandb.log({"epoch": epoch,
#     #  "Query set per loss": query_set_perloss, 
#     #  "Original train per loss": original_train_perloss,
#     #  "cal_loss":0.5*NDloss_total + 0.25*EDloss_total})

#     return cal_loss

def fast_adapt_unlearn(args, epoch, batch1, batch2, model, shots, trainer, tester, save_path):
    EDloss_total=0
    NDloss_total=0
    shots=0
    log_grad_module(model,save_path,"grad_when_unlearn")

    for AD in batch1:
        ED_loss, ED_loss_dict = trainer(model,AD,'image','txt',-(len(AD['image'])-shots),sample=args.sample)
        if len(args.gpus.split(","))>1:
            EDloss_total+=sum(ED_loss_dict['black_loss'])
        else:
            # import pdb;pdb.set_trace()
            EDloss_total+=ED_loss_dict['black_loss']
    for ND in batch2:
        ND_loss, ND_loss_dict = trainer(model,ND,'image','txt',None,sample="random")
        if len(args.gpus.split(","))>1:
            NDloss_total+=sum(ND_loss)
        else:
            NDloss_total+=ND_loss


    EDloss_total /= len(batch1)
    NDloss_total /= len(batch2)
    query_set_perloss = EDloss_total.item()
    original_train_perloss = NDloss_total.item()
    cal_loss = args.alpha*NDloss_total + args.beta*EDloss_total
    # cal_loss = args.alpha*NDloss_total 
    print('Query set per loss', query_set_perloss)
    print(f'Original train per loss {original_train_perloss}')
    # wandb.log({"ED_loss":query_set_perloss,"ND_loss":original_train_perloss,"cal_loss":cal_loss})
    with open(f'{save_path}/log.txt', 'a') as file:
        file.write(f"epoch:{epoch},ED_loss:{query_set_perloss},ND_loss: {original_train_perloss},cal_loss:{cal_loss}\n")
    # wandb.log({"epoch": epoch,
    #  "Query set per loss": query_set_perloss, 
    #  "Original train per loss": original_train_perloss,
    #  "cal_loss":0.5*NDloss_total + 0.25*EDloss_total})

    return cal_loss


def fine_tune(args, epoch, batch1, batch2, model, shots, trainer, tester, save_path):
    ADloss_total=0
    EDloss_total=0
    NDloss_total=0
    if args.strategy=="unlearn" or "FAAT":
        shots=0
    # import pdb;pdb.set_trace()
    # unet_and_lora_require_grad(model,enable=True)

    for AD in batch1:
        ED_loss, ED_loss_dict = trainer(model,AD,'image','txt',-(len(AD['image'])-shots))
        EDloss_total+=ED_loss
    

    EDloss_total /= len(batch1)
    NDloss_total /= len(batch2)
    query_set_perloss = EDloss_total.item()
    original_train_perloss = NDloss_total
    cal_loss = args.alpha*NDloss_total + args.beta*EDloss_total
    # cal_loss = args.alpha*NDloss_total 
    print('Query set per loss', query_set_perloss)
    print(f'Original train per loss {original_train_perloss}')
    with open(f'{save_path}/log.txt', 'a') as file: 
        file.write(f"epoch:{epoch},ED_loss:{query_set_perloss},ND_loss: {original_train_perloss},cal_loss:{cal_loss}\n") 
    # wandb.log({"epoch": epoch,
    #  "Query set per loss": query_set_perloss, 
    #  "Original train per loss": original_train_perloss,
    #  "cal_loss":0.5*NDloss_total + 0.25*EDloss_total})

    return cal_loss