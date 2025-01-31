import os
import argparse
# import pdb
from tqdm import tqdm
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-bs', '--batch-size', default=100, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--fast_lr', default=0.01, type=float)
    parser.add_argument('--alpha', default=10.0, type=float, help='coefficient of maml lr')
    parser.add_argument('--beta', default=1.0, type=float, help='coefficient of natural lr')
    parser.add_argument('--cigma', default=1.0, type=float, help='coefficient of feature supervise')
    parser.add_argument('--gamma', default=1.0, type=float, help='coefficient of feature supervise')
    parser.add_argument('--finetune_lr', default=0.0001, type=float)
    parser.add_argument('--finetune_number', default=500, type=int)
    parser.add_argument('--shots', default=10, type=int)
    parser.add_argument('--test_iterval', default=100, type=int)
    parser.add_argument('--root', default='./results_patronus', type=str)                      
    parser.add_argument('--finetune_epochs', default=1, type=int)
    parser.add_argument('--weighted', default=1, type=int, help='whether weighted during dos and inverse loss')
    parser.add_argument('--coefficients', default='1,0,0', type=str, help='coefficients among dos loss, inverse loss, gradient loss')
    parser.add_argument('--batches', default=50, type=int)
    parser.add_argument('--target', default='celeba', type=str, choices=['porn', 'sexy', 'gunset'])
    parser.add_argument('--maml', default='m', type=str, choices=['s', 'm', 'a', 'x'])
    parser.add_argument('--original_train_list', default=None, type=str)
    parser.add_argument('--original_test_list', default=None, type=str)
    parser.add_argument('--target_test_list', default=None, type=str)
    parser.add_argument('--target_defense_list', default=None, type=str)
    parser.add_argument('--target_finetune_list', default=None, type=str)
    parser.add_argument('--truly_finetune_freq', default=1000, type=int)
    parser.add_argument('--truly_finetune_epochs', default=20, type=int)
    parser.add_argument('--gpus', default=None, type=str)
    parser.add_argument('--same_path', default=0, type=int)
    parser.add_argument('--mixed', default=1, type=int)
    parser.add_argument('--partial', default=0, type=int)
    parser.add_argument('--pretest', default=0, type=int)
    parser.add_argument('--only_decoder', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--syn', default=0, type=int)
    parser.add_argument('--disalbe_wandb', default=1, type=int)
    parser.add_argument('--unlearn', default=0, type=int)
    parser.add_argument('--x', default=0, type=int)
    parser.add_argument('--loss', default=None, type=str)
    parser.add_argument('--arch', default='32x4', type=str)
    parser.add_argument('--commit', default=None, type=str)
    parser.add_argument('--resume_ckpt', default=None, type=str)
    parser.add_argument('--syn_original_train_list', default=None, type=str)
    parser.add_argument('--syn_original_test_list', default=None, type=str)
    parser.add_argument('--syn_target_defense_list', default=None, type=str)
    parser.add_argument('--syn_target_finetune_list', default=None, type=str)
    parser.add_argument('--target_feature_dir', default=None, type=str)
    parser.add_argument('--syn_target_test_list', default=None, type=str)
    parser.add_argument('--mimic', default=0, type=int)
    parser.add_argument('--resolution', default=224, type=int)
    parser.add_argument('--threshold', default=3, type=int)
    args = parser.parse_args()
    if args.target == 'porn':
        args.target_feature_dir =  './target_features/pornfeatures/feature'
    elif args.target == 'gunset':
        args.target_feature_dir = './target_features/gunpromptfeatures/feature'
    elif args.target == 'sexy':
        args.target_feature_dir = './target_features/sexypromptfeatures/feature'
    else:
        raise NotImplementedError
    return args

args = get_args()
if args.gpus:
    gpu_list = args.gpus.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    devices_id = [id for id in range(len(gpu_list))]
import torch
import numpy as np
import copy
from datetime import datetime
import wandb
from tools.image_loss import vggloss
import os
import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
from utils import set_seed, save_args_to_file, test, test_finetune, draw, get_dataset, get_features, homemade_test_finetune, get_featureset, weighted

def main(args):
    names = args.name if args.name is not None else f'maml{args.maml}_alpha{args.alpha}_beta{args.beta}'
    if not args.disalbe_wandb:
        wandb.init(
        project = "new t2i gogo",  
        entity = "pangpang",
        config = args,
        name = names,
        notes = args.notes,
        # mode="offline"         
    
    )   
    set_seed(args.seed)
    save_path = args.root + '/'
    now = datetime.now()
    save_path = save_path + '/' + f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'
    if args.resume_ckpt and args.same_path:
        print(f'Resuming from {args.resume_ckpt}')
        index = args.resume_ckpt.find("ckpts")
        if index >0:
            resumed_save_path = args.resume_ckpt[:index]
            save_path = resumed_save_path
            print(f'Save to {save_path}...continue training...')
        save_args_to_file(args, save_path+"newargs.json")
            
    else:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'/ckpts/', exist_ok=True)
        os.makedirs(save_path+'/finetuned_ckpts/', exist_ok=True)
        save_args_to_file(args, save_path+"args.json")
    if not args.disalbe_wandb:
        wandb.log({'save path': save_path})
        wandb.config.update(args)
    
        log_dir = wandb.run.dir
        if os.path.exists(save_path+"/wandb_log_dir.txt"):
            new_txt_file_path = save_path+"/new_wandb_log_dir.txt"
            with open(new_txt_file_path, "w") as f:
                f.write('wandb sync '+ log_dir[:-6] + '\n')
                f.write('git reset --hard ' + args.commit)
        else:
            txt_file_path = save_path+"/wandb_log_dir.txt"
            with open(txt_file_path, "w") as f:
                f.write('wandb sync '+ log_dir[:-6] + '\n')
                f.write('git reset --hard ' + args.commit)
        
    from models.vae import AutoencoderKL, ddconfig32, ddconfig64, Decoder
    ddconfig = ddconfig32 if args.arch == '32x4' else ddconfig64
    embed_dim = 4 if args.arch == '32x4' else 3
    ddconfig['resolution'] = args.resolution
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=embed_dim)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    if args.maml == 's':
        from maml.originalMAML import MAML
    elif args.maml == 'm':
        from maml.momentumMAML import MAML
    elif args.maml == 'a':
        from maml.adamMAML import MAML
    elif args.maml == 'x':
        from maml.mixedMAML import MAML
    maml = MAML(model, lr=args.fast_lr, first_order=True)
    opt1 = optim.Adam(maml.parameters(), args.lr, weight_decay=args.wd)
    # opt2 = adai.AdaiV2(maml.parameters(), args.lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=args.wd, decoupled=False)

    print('=> loading ckpt ...')
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt['state_dict'])
        if 'loop' in ckpt.keys():
            if args.same_path:
                args.start_epoch = ckpt['loop']
    ## load original model parameter as teacher model
    teacher_model = copy.deepcopy(model)
    ckpt = torch.load('./assets/pretrained32x32x4.pt')
    teacher_model.load_state_dict(ckpt['state_dict'])   
    
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    print('=> building the dataloader ...')
    target_loader, target_finetuneloader, target_testloader, original_loader, original_testloader, synthesized = get_dataset(args)
    # import pdb;pdb.set_trace()
    features = get_features(['./normalfeatures/feature', './2000originalfeatures/feature'], [args.target_feature_dir])
    normal_trainloader, normal_testloader, porn_trainloader, porn_testloader = get_featureset(features['normal'], features['porn'],4)

    # import pdb
    # pdb.set_trace()
    print('preparing iteration...')
    target_iter = iter(target_loader)
    original_iter = iter(original_loader)
    normal_feature_iter = iter(normal_trainloader)
    porn_feature_iter = iter(porn_trainloader)
    porn_test_iter = iter(porn_testloader)
    print(f'===> length of dataloader: target_iter:{len(target_loader)}, original_iter:{len(original_loader)}')


    if args.same_path:
        print('Jumping to the resume iteration...')
        for _ in tqdm(range(args.start_epoch%(int(len(target_loader)/args.batches)))):
            try:
                _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
        for _ in tqdm(range(args.start_epoch%len(original_loader))):
            try:
                _ = next(original_iter)
            except StopIteration:
                original_iter = iter(original_loader)
            
    print('=> building the criterion ...')
    mse_criterion = nn.MSELoss()
    model.train()
    print('=> starting training engine ...')
    best = 100
    
    patronus_criterion = vggloss
    
    if args.mimic:
        # fastadapt = patronus
        from patronus_losses import single_patronus, mixed_patronus
        if args.mixed:
            fastadapt = mixed_patronus
        else:
            print('---Using single patronus---')
            fastadapt = single_patronus
    else:
        if args.partial:
        
            from patronus_losses import partial_patronus
            fastadapt = partial_patronus
        else:
            from patronus_losses import patronus
            fastadapt = patronus
        
    if args.only_decoder:
        print('Freeze Encoder......')
        # for name, module in model.named_parameters():
        #     if 'encoder' in name:
        #         module.requires_grad = False
        for name, module in model.named_parameters():
                module.requires_grad = False
        for name, module in model.named_parameters():
            if 'decoder' in name:
                module.requires_grad = True
            if 'post_quant_conv' in name:
                module.requires_grad = True
    if not args.disalbe_wandb: 
        define_wandb()
    if args.pretest:
        print('\n*************Pre test**************')
        print('       ----test finetune----')
        try:
            porn_features = next(porn_feature_iter)
        except StopIteration:
            porn_feature_iter = iter(porn_trainloader)
            porn_features = next(porn_feature_iter)
        porn_features = torch.squeeze(porn_features).cuda()
        porn_features = 1. / 0.18215 * porn_features
        finetunetest_loss, finetunefeature_loss, outputs, feature_output, testmodel = test_finetune(copy.deepcopy(model.module), target_finetuneloader, target_testloader, args.finetune_epochs, args, porn_features)
        draw(-1, finetunetest_loss, outputs, save_path + f'/rec/pretest/target', args.resolution)
        draw(-1, finetunefeature_loss, feature_output, save_path + f'/rec/pretest/target_feature', args.resolution)
        print(f'finetune performance: test loss is {finetunetest_loss}')  
        if not args.disalbe_wandb:           
            wandb.log({"Pretest/Finetune performance test loss":finetunetest_loss})
        print('       ----test original----')
        original_testloss, outputs = test(copy.deepcopy(model.module), original_testloader, torch.device('cuda'))
        draw(-1, original_testloss, outputs, save_path + f'/rec/pretest/original', args.resolution)
        print(f'original performance: test loss is{original_testloss}')  
        if not args.disalbe_wandb:        
            wandb.log({"Pretest/Original performance test loss":original_testloss})
    threshold = args.threshold
    adapt_method = False
    for total_loop in range(args.start_epoch, args.epochs):
        if not args.disalbe_wandb:        
            wandb.log({'epoch':total_loop})
        print(f'\n================LOOP {total_loop}=================')
        test_index = 0
        backup = copy.deepcopy(model)
        opt1.zero_grad()
        # opt2.zero_grad()
        batches = []
        feature_batches = []
        for _ in range(args.batches):
            try:
                batch = next(target_iter)
                batches.append(batch)
            except StopIteration:
                target_iter = iter(target_loader)
                batch = next(target_iter)
                batches.append(batch)

        for _ in range(args.batches):
                
            try:
                porn_features = next(porn_feature_iter)
            except StopIteration:
                porn_feature_iter = iter(porn_trainloader)
                porn_features = next(porn_feature_iter)
            porn_features = torch.squeeze(porn_features).cuda()
            porn_features = 1. / 0.18215 * porn_features
            feature_batches.append(porn_features)
            
        learner = maml.clone()
        if total_loop%1 == 0:
            adapt_method = not adapt_method
        # import pdb;pdb.set_trace()
        adapt_method, patronus_loss, porn_feature_loss, inverse_loss, gradient_loss, evaluation_predictions, reconstruction_loss, _ = fastadapt(batches, learner, patronus_criterion, threshold, torch.device('cuda'), feature_batches, adapt_method, teacher_model) 
        
        if (total_loop+1) % args.test_iterval == 0:
            draw(total_loop, reconstruction_loss, evaluation_predictions, save_path + f'/rec/target_trained/', args.resolution)
            # draw(total_loop, -1, targets, save_path + f'/rec/destroyed_targets/')
        if not args.disalbe_wandb:
            if adapt_method:
                wandb.log({"patronus train loss of momentum maml": patronus_loss.item(),})
            else:
                wandb.log({"patronus train loss of adam maml": patronus_loss.item(),})
            
        try:
            batch = next(original_iter)
        except StopIteration:
            original_iter = iter(original_loader)
            batch = next(original_iter)
        inputs = batch
        targets = inputs
        inputs, targets = inputs.cuda(), targets.cuda()   
        # with autocast():
        outputs = model(inputs)[0]
        natural_loss = mse_criterion(outputs, targets) 
        if not args.disalbe_wandb:
            wandb.log({"Original train loss": natural_loss.item(),})
        
        
        # #### feature supervise
        model_output = torch.clamp((model.module.decode(porn_features) + 1.0) / 2.0, min=0.0, max=1.0)
        dos_loss = patronus_criterion(model_output, torch.zeros_like(model_output))
        if not args.disalbe_wandb:
            wandb.log({"Dos feature loss": dos_loss.item(),})
        dos_loss += porn_feature_loss
        
        if (total_loop+1) % args.test_iterval == 0:
            draw(total_loop, -1, model_output, save_path + f'/rec/trained_feature_target/', args.resolution)

        try:
            features = next(normal_feature_iter)
        except StopIteration:
            normal_feature_iter = iter(normal_trainloader)
            features = next(normal_feature_iter)
        features = torch.squeeze(features).cuda()
        features = 1. / 0.18215 * features 
        
        teacher_output = torch.clamp((teacher_model.module.decode(features) + 1.0) / 2.0, min=0.0, max=1.0)
        model_output = torch.clamp((model.module.decode(features) + 1.0) / 2.0, min=0.0, max=1.0)
        feature_loss = mse_criterion(teacher_output, model_output) 
        if not args.disalbe_wandb:
            wandb.log({"Normal feature loss": feature_loss.item(),})
        
        if (total_loop+1) % args.test_iterval == 0:
            draw(total_loop, -1, model_output, save_path + f'/rec/trained_feature_original/', args.resolution)
            draw(total_loop, -1, teacher_output, save_path + f'/rec/teacher_feature_original/', args.resolution)
        
        print(f'Combined loss = patronus loss + natural loss + dos feature loss + original feature loss')
        print(f'losses are: {round(patronus_loss.item(),5)} {round(natural_loss.item(),5)} {round(dos_loss.item(),5)} {round(feature_loss.item(),5)}')

        nash_weights = 0
        if nash_weights:
            weights, anti_loss, intact_loss  = weighted(args, model, patronus_loss, natural_loss, dos_loss, feature_loss)
            combined_loss = np.clip(np.array(weights['ce1']),0.1,0.9)*anti_loss + np.clip(np.array(weights['ce2']),0.1,0.9)*intact_loss
            print(f'merged losses are anti loss + intact loss: {round(anti_loss.item(),5)} {round(intact_loss.item(),5)}')
            print('weights are:', round(weights['ce1'],4),round(weights['ce2'],4))

        else:
            combined_loss = args.alpha*patronus_loss + args.beta*natural_loss + args.cigma*dos_loss + args.gamma*feature_loss
            print('weights are:', args.alpha, args.beta, args.cigma, args.gamma)

        

        combined_loss.backward()
        opt1.step()
            
        #check for freeze
        flag1 = 1
        flag2 = 1
        for (name1, param1), (name2, param2) in zip(backup.named_parameters(), model.named_parameters()):
            if ('encoder' in name1) and (not torch.equal(param1, param2)):
                flag1 = 0
            if ('decoder' in name1) and (not torch.equal(param1, param2)):
                flag2 = 0
        if flag1:
            pass
        else: 
            assert(0)

        if (total_loop+1) % args.test_iterval == 0:
            test_index += 1
            if not args.disalbe_wandb:
                wandb.log({'test_index': test_index})
            # torch.cuda.empty_cache()
            print('\n*************test finetuned model**************')
            
            print('          ----test finetune----')
            test_model = copy.deepcopy(model.module)
            #finetune_epochs=10 when total_loop==999 or 1999
            finetune_epochs = args.truly_finetune_epochs if (total_loop+1) % args.truly_finetune_freq == 0 else args.finetune_epochs
            
            #test momentum finetune
            try:
                porn_testfeatures = next(porn_test_iter)
            except StopIteration:
                porn_feature_iter = iter(porn_testloader)
                porn_testfeatures = next(porn_test_iter)
                porn_testfeatures = torch.squeeze(porn_testfeatures).cuda()
            porn_testfeatures = 1. / 0.18215 * porn_testfeatures
            finetunetest_loss, finetunefeature_loss, outputs, feature_output, testmodel = test_finetune(test_model, target_finetuneloader, target_testloader, finetune_epochs, args, porn_testfeatures)
            print(f' Momentum Finetune: unlearning performance: recon loss is {finetunetest_loss}, feature loss is {finetunefeature_loss} @ {finetune_epochs} epoch finetune')  
                        
            if (total_loop+1) % args.truly_finetune_freq == 0:
                name1 = f'/rec/finetuned_target_test_momentum/'
                name2 = f'/rec/finetuned_target_feature_test_momentum/'
            else:
                name1 = f'/rec/target_test_momentum/'
                name2 = f'/rec/target_feature_test_momentum/'
            draw(total_loop, finetunetest_loss, outputs, save_path + name1, args.resolution)
            draw(total_loop, finetunefeature_loss, feature_output, save_path + name2, args.resolution)
            if not args.disalbe_wandb:
                wandb.log({"momentum Finetune performance test loss":finetunetest_loss})
            
            #test adam finetune
            # finetunetest_loss, finetunefeature_loss, outputs, feature_output, testmodel = homemade_test_finetune(test_model, target_finetuneloader, target_testloader, finetune_epochs, args, porn_testfeatures)
            # if (total_loop+1) % args.truly_finetune_freq == 0:
            #     name1 = f'/rec/finetuned_target_test_adam/'
            #     name2 = f'/rec/finetuned_target_feature_test_adam/'
            # else:
            #     name1 = f'/rec/target_test_adam/'
            #     name2 = f'/rec/target_feature_test_adam/'
            # draw(total_loop, finetunetest_loss, outputs, save_path + name1, args.resolution)
            # draw(total_loop, finetunefeature_loss, feature_output, save_path + name2, args.resolution)
            # if not args.disalbe_wandb:
            #     wandb.log({"adam Finetune performance test loss":finetunetest_loss})
            
            torch.save({
            'state_dict':testmodel.state_dict(),
            },save_path+f'/finetuned_ckpts/{total_loop}.pt')
            print(f' Adam Finetune: unlearning performance: recon loss is {finetunetest_loss}, feature loss is {finetunefeature_loss} @ {finetune_epochs} epoch finetune')  


            
            print('          ----test original----')
            original_testloss, outputs = test(copy.deepcopy(model.module), original_testloader, torch.device('cuda'))
            draw(total_loop, original_testloss, outputs, save_path + f'/rec/original_test/', args.resolution)
            print(f'original performance: test loss is{original_testloss}')  
            if not args.disalbe_wandb:
                wandb.log({"TestTime/Original performance test loss":original_testloss})
           
            name = f'loop{total_loop}_ori{round(original_testloss,3)}_ft{round(finetunetest_loss,3)}_qloss{round(patronus_loss.item()/args.alpha,5)}.pt'
            torch.save({
            'state_dict':model.state_dict(),
            'loop':total_loop},save_path+'/ckpts/'+name)
            gain = -finetunetest_loss + finetunefeature_loss + original_testloss
            if gain < best:
                best = gain
                torch.save({'state_dict':model.state_dict(),
                            'loop':total_loop,
                            },save_path+'/ckpts/'+ f'best@loop_{total_loop}@tarloss{round(finetunetest_loss,3)}.pt')
                
            print('************************************************')

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


