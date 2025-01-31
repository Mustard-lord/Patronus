import os
import sys
import random
import numpy as np
import torch.nn as nn
from loguru import logger
import json
from torchvision.transforms import transforms
from tools.MGDA import MGDASolver
from learn2learn.utils import update_module
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
import pdb
import wandb
from torchvision.transforms import transforms
from torch.backends import cudnn
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
import torch.utils.data as data

def get_features(normal_path, porn_path, num=None):
    total_features = {}
    for name, paths in zip(['normal','porn'],[normal_path, porn_path]):
        features = []
        for path in paths:
         for pt in os.listdir(path):
            features += torch.load(os.path.join(path,pt))
        features = torch.stack(features)
        total_features[name] = features
    return total_features


class ImageDataset(data.Dataset):
    def __init__(self, ann_files, num=None, transform=None, target_noise=None, suffix=None):
        self.ann_file = ann_files
        self.transform = transform
        self.num = num
        self.target_noise = target_noise
        self.init()
        if target_noise == 'blur':
            # self.blurer = GaussianBlur(9,1)
            self.blurer = transforms.GaussianBlur(91,sigma=(50,150))
        if suffix is not None:
            newfiles = [file[:-8]+suffix+'_list.txt' for file in ann_files]
            self.ann_file = ann_files
            

    def init(self):
        # pdb.set_trace()
        self.im_names = []
        self.noises = []
        for ann_file in self.ann_file:
            if ann_file:
                print(f'==>loading from {ann_file}')
                with open(ann_file, 'r') as f:
                        lines = f.readlines()
                        progress_bar = tqdm(total=len(lines)) 
                        for line in lines:
                            progress_bar.update(1)
                            data = line.strip().split(' ')
                            self.im_names.append(data[0])
                            # self.noises.append(torch.normal(mean=0, std=1, size=[3,224,224]))
        if self.num:
            self.im_names = self.im_names[:self.num]
            # self.noises = self.noises[:self.num]
        print(f'dataset length:{len(self.im_names)} \n')
        
    def __getitem__(self, index):
        im_name = self.im_names[index]
        img = Image.open(im_name).convert('RGB') 
        img = self.transform(img)
        
        if self.target_noise:
            # if self.target_noise == 'normal':
            #     target = self.noises[index]
            if self.target_noise == 'zero':
                target = torch.zeros_like(img)
            elif self.target_noise == 'blur':
                target = self.blurer(img)

            else:
                NotImplementedError
                
        else:
            target = img

        return img

    def __len__(self):
        return len(self.im_names)

def get_dataset(args):

    target_defense_loader = load_dataset(args, args.batch_size, args.target_defense_list.split(','), target_noise=args.loss, train=1, suffix=0)
    target_finetuneloader = load_dataset(args, args.batch_size, args.target_finetune_list.split(','), num=args.finetune_number, train=1)
    target_testloader = load_dataset(args, 1, args.target_test_list.split(','), train=0)
    original_testloader = load_dataset(args, 20, args.original_test_list.split(','), train=0)
    synset= {}
    original_loader = load_dataset(args, args.batch_size, args.original_train_list.split(','), train=1)
        
    return target_defense_loader, target_finetuneloader, target_testloader, original_loader, original_testloader, synset

    
class RandomHorizontalFlip:
    def __call__(self, sample):
        image = sample
        if np.random.rand() > 0.5:
            image = image.flip(-1)
        return image
    
def get_featureset(normal_features, porn_features,bs):
    aug = transforms.Compose([
    RandomHorizontalFlip(),  
    # RandomRotation(degrees=30),
    # AddGaussianNoise(mean=0.0, std=0.01)         
])
    normalset = AugmentedFeatureDataset(normal_features, transform=aug)
    pornset = AugmentedFeatureDataset(porn_features, transform=aug)
    normal_train, normal_test = data.random_split(normalset, [0.8, 0.2])
    porn_train, porn_test = data.random_split(pornset, [0.8, 0.2])
    normal_train_loader = DataLoader(normal_train, batch_size=4, num_workers=0)
    porn_train_loader = DataLoader(porn_train, batch_size=4, num_workers=0)
    normal_test_loader = DataLoader(normal_train, batch_size=4, num_workers=0)
    porn_test_loader = DataLoader(porn_train, batch_size=4, num_workers=0)
    return normal_train_loader, normal_test_loader, porn_train_loader, porn_test_loader



class AugmentedFeatureDataset(data.Dataset):
    def __init__(self, features, transform=None):
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return torch.unsqueeze(feature,0)
    
def get_grads(net, loss):
    # pdb.set_trace()
    params = [x for x in net.parameters() if x.requires_grad]
    grads = list(torch.autograd.grad(loss, params, retain_graph=True))
    return grads  
   
def draw(epoch, loss, output, path, resolution=256, counter=-1):
    # print(resolution)
    # print(output.shape)
    if counter != -1:
        path = path + f'/{counter}/'
    os.makedirs(path, exist_ok=True)
    canvas_width = 5 * resolution
    canvas_height = 4 * resolution
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    canvas_image = Image.fromarray(canvas)

    for i in range(output.size(0)):
        img = output[i]
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).astype('uint8')
        row = i // 5  
        col = i % 5   
        x_offset = col * resolution  
        y_offset = row * resolution 
        canvas_image.paste(Image.fromarray(img), (x_offset, y_offset))
    canvas_image.save(path + f"/{epoch}loop@loss_{loss}.png")
    # wandb.log({"test img": canvas_image})
    # print(f'Saving generated images in {path}')

def maml_update(m, v, model, lr, grads=None):
    if m is None:
        m, v = {}, {}
        for name, param in model.named_parameters():
            m[name] = torch.zeros_like(param)
            v[name] = torch.zeros_like(param)
    for grad_counter, (key, param) in enumerate(model.named_parameters()):
        if grads[grad_counter] is None:
            continue
        m[key] = 0.9*m[key] + (1-0.9)*grads[grad_counter]
        v[key] = 0.999*v[key] + (1-0.999)*(grads[grad_counter]**2)
        update = -lr*m[key] / (torch.sqrt(v[key]) + 1e-8)
        param.update = update
    return update_module(model), m, v

def adapt(  lr,
            iteration,
            model,
            loss,
            m=None,
            v=None,
            first_order=None,
            allow_unused=None,
            ):
    diff_params = [p for p in model.parameters() if p.requires_grad]
    grad_params = grad(loss,
                        diff_params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True)
    gradients = []
    grad_counter = 0

    for param in model.parameters():
        if param.requires_grad:
            gradient = grad_params[grad_counter]
            grad_counter += 1
        else:
            gradient = None
        gradients.append(gradient)

    # Update the module
    lr_t = lr * np.sqrt(1.0 - 0.999**iteration) / (1.0 - 0.9**iteration)
    return maml_update(m, v, model, lr_t, gradients)
    

def homemade_test_finetune(model, tar_finetuneloader, tar_testloader, epochs, args, porn_features):
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    test_loss = -1
    m = None
    v = None
    model_output = torch.zeros(2,3,256,256)
    dos_loss = torch.tensor(-1)
    
    print('\nTest adam finetune')
    for ep in range(epochs):
        model.train()
        for index, inputs in enumerate(tar_finetuneloader):
            # print(f'Test adam finetune - Number Iteration {index}')
            if index == 20:
                    break
            inputs = inputs.cuda()
            outputs = model(inputs)[0]
            loss = criterion(outputs, inputs)
            model, m, v = adapt(1e-4, index+1, model, loss, m ,v)
            test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
            model_output = torch.clamp((model.module.decode(porn_features) + 1.0) / 2.0, min=0.0, max=1.0)
            dos_loss = criterion(model_output, torch.zeros_like(model_output))
        if epochs != 1:
            print('finetuned test loss:', round(test_loss,5))
            if not args.disalbe_wandb:
                wandb.log({'truly_finetune_epoch': ep})
                wandb.log({"Truly Finetune/performance test loss":test_loss})
    test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
    print('vae test reconstruction loss:',round(test_loss,5))
    print('feature dos zero loss:', round(dos_loss.item(),4))
    return round(test_loss,7), round(dos_loss.item(),7), recons, model_output, model

def weighted(args, model, patronus_loss, natural_loss, dos_loss, feature_loss):
        anti_loss = patronus_loss + dos_loss
        intact_loss = natural_loss + feature_loss
        anti_grad = get_grads(model, anti_loss)
        intact_grad = get_grads(model, intact_loss)
        scales = MGDASolver.get_scales(dict(ce1 = anti_grad, ce2 = intact_grad),
                                       dict(ce1 = anti_loss, ce2 = intact_loss),
                                       'none' , ['ce1', 'ce2'])
        return scales, anti_loss, intact_loss
   
def test_finetune(model, tar_finetuneloader, tar_testloader, epochs, args, porn_features):
    model = nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), args.finetune_lr, momentum=0.9, weight_decay=1e-4)
    
    # if args.maml == 'm':
    #     optimizer = optim.SGD(model.parameters(), args.finetune_lr, momentum=0.9, weight_decay=1e-4)
    # elif args.maml == 'a':
    #     optimizer = optim.Adam(model.parameters(), args.finetune_lr, weight_decay=0)
    dos_loss = torch.tensor(-1)
    criterion = nn.MSELoss()
    test_loss = -1
    model_output = torch.zeros(2,3,256,256)
    if not args.disalbe_wandb:    
        wandb.define_metric("truly_finetune_epoch")
        wandb.define_metric("Truly Finetune/performance test loss", step_metric='truly_finetune_epoch')
    # pdb.set_trace()
    print("Test monmentum finetune")
    for ep in tqdm(range(epochs)):
        model.train()
        for index, inputs in tqdm(enumerate(tar_finetuneloader)):
            if args.maml == 'a' and index == 20:
                    break
            inputs = inputs.cuda()
            outputs = model(inputs)[0]
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
        test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
        print('vae test reconstruction loss:',round(test_loss,5))
        model_output = torch.clamp((model.module.decode(torch.squeeze(porn_features)) + 1.0) / 2.0, min=0.0, max=1.0)
        dos_loss = criterion(model_output, torch.zeros_like(model_output))
        print('feature dos zero loss:', round(dos_loss.item(),4))
        if epochs != 1:
            print('finetuned test loss:', round(test_loss,5))
            if not args.disalbe_wandb:
                wandb.log({'truly_finetune_epoch': ep})
                wandb.log({"Truly Finetune/performance test loss":test_loss})
    test_loss, recons = test(model, tar_testloader, torch.device('cuda'))
    return round(test_loss,7), round(dos_loss.item(),7), recons, model_output, model

def test(model, original_testloader, device):
    test_loss = 0
    total = 0
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(original_testloader):
            if batch_idx == 10:
                break
            inputs = inputs.to(device)
            outputs = model(inputs)[0]
            loss = criterion(outputs, inputs) 
            test_loss += loss.item() * outputs.shape[0]
            if batch_idx == 0:
                recons = outputs
            total += inputs.size(0)
    return test_loss*1.0/total, recons


def load_dataset(args, bs, train_lists, num=None, target_noise=None, train=0, suffix=0):
    print(f'Change target to {target_noise}')
    if train:
        augmentation = [
        transforms.Resize([args.resolution,args.resolution]),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(90),
        # transforms.ColorJitter(10),
        transforms.ToTensor(),
        ]
    else:
        augmentation = [
        transforms.Resize([args.resolution,args.resolution]),
        transforms.ToTensor(),
        ]
        
    trans = transforms.Compose(augmentation)
    # pdb.set_trace()
    if suffix:
        adapt_dataset = ImageDataset(train_lists, num=num, transform=trans, target_noise=target_noise, suffix='adapt')
        eval_dataset = ImageDataset(train_lists, num=num, transform=trans, target_noise=target_noise,suffix='eval')
        
        adapt_loader = torch.utils.data.DataLoader(
                    adapt_dataset,
                    shuffle=True,
                    batch_size=bs,
                    num_workers=40,
                    pin_memory=False,
                    drop_last=False)
        eval_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    shuffle=True,
                    batch_size=bs,
                    num_workers=40,
                    pin_memory=False,
                    drop_last=False)
        return adapt_loader, eval_loader
   
    else:
        train_dataset = ImageDataset(train_lists, num=num, transform=trans, target_noise=target_noise)   
        
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=bs,
                        num_workers=40,
                        pin_memory=False,
                        drop_last=False)
        return train_loader

def save_args_to_file(args, file_path):
    with open(file_path, "w") as file:
        json.dump(vars(args), file, indent=4)

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

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    decay = args.lr_drop_ratio if epoch in args.lr_drop_epoch else 1.0
    lr = args.lr * decay
    global current_lr
    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.lr = current_lr
    return current_lr


def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True