import torch
import sys
sys.path.append("..")
from ldm.util import instantiate_from_config
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
import time
import torchvision
import pytorch_lightning as pl
import os
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from ldm.data.base import Txt2ImgIterableBaseDataset
import numpy as np
import json
from tools.param_ctrl import unet_and_lora_require_grad,log_grad_module
from tools.draw import draw_img,draw_loss_curves,plot_FT_loss
from torch.optim.lr_scheduler import StepLR


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
    
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train_porn=None,train_norm=None, target_finetune=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_val_dataloader=shuffle_val_dataloader
        self.shuffle_test_loader=shuffle_test_loader
        # self.setup()
        if train_porn is not None:
            self.dataset_configs["train_porn"] = train_porn
        
        if train_norm is not None:
            self.dataset_configs["train_norm"] = train_norm
            
        if target_finetune is not None:
            self.dataset_configs["target_finetune"] = target_finetune   

            # self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            # self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            # self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            # self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        if "train_porn" in self.datasets.keys() and "train_norm" in self.datasets.keys():
            self.train_porn_dataloader,self.train_norm_dataloader = self._train_dataloader()
        if "target_finetune" in self.datasets.keys():
            self.target_finetuneloader = self._target_ft_dataloader()
        if "validation" in self.datasets.keys():
            self.val_dataloader = partial(self._val_dataloader, shuffle=self.shuffle_val_dataloader)
        if "test" in self.datasets.keys():
            self.test_dataloader = partial(self._test_dataloader, shuffle=self.shuffle_test_loader)
        if "predict" in self.datasets.keys():
            self.predict_dataloader = self._predict_dataloader()

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train_porn'], Txt2ImgIterableBaseDataset) or isinstance(self.datasets['train_norm'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return (DataLoader(self.datasets["train_porn"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn),
                DataLoader(self.datasets["train_norm"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn))
        
    def _target_ft_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['target_finetune'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["target_finetune"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)
    
    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


def load_model_from_config(config, ckpt, verbose=False, myvae=None):
    print(f"Loading model from {ckpt}")
    # import pdb;pdb.set_trace()
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model.cuda()

def save_args_to_json(args, config, filename,**kwarg):
    arg_dict = vars(args)
    dictconfig_dict = OmegaConf.to_container(config, resolve=True)
    # if kwarg != None:
    total_args = {**arg_dict, **dictconfig_dict,**kwarg}
    # else:
    #     total_args = {**arg_dict, **dictconfig_dict}
    json_str = json.dumps(total_args, indent=4)

    with open(filename, 'w') as jsonfile:
        jsonfile.write(json_str)


                                        
# def test_finetune(model, trainer, tar_finetuneloader, epoch):
#     unet_and_lora_require_grad(model,enable=True)
#     optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
#     print("Test monmentum finetune")

#     for ep in tqdm(range(epoch)):
#         model.train()
#         for index, inputs in tqdm(enumerate(tar_finetuneloader)):
#             loss, _ = trainer(model,inputs,'image','txt',None)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
     
#     return model

def test_finetune(model, trainer, tar_finetuneloader, lr, optimizer_select, epoch,save_path,img_store_path,prompt_paths,resume):
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    
    # if optimizer_select == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # elif optimizer_select == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # elif optimizer_select == 'adade':
    #     optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-6)
    # elif optimizer_select == 'rms':
    #     optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.0)
    # elif optimizer_select == 'nes':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    if optimizer_select == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_select == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    elif optimizer_select == 'adade':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, rho=0.9, eps=1e-6)
    elif optimizer_select == 'rms':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.0)
    elif optimizer_select == 'nes':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, nesterov=True)
    print("Test monmentum finetune")
    scheduler = StepLR(optimizer, step_size=len(tar_finetuneloader)*epoch//4, gamma=0.1)
    
    count=0
    for ep in tqdm(range(epoch)):
        model.train()
        print(f"******************epoch:{ep}************************\n")
        for index, inputs in tqdm(enumerate(tar_finetuneloader)):
            if resume!= None and count<resume:
                count=count+1
                continue
            optimizer.zero_grad()
            loss, _ = trainer(model,inputs,'image','txt',None,sample="random")
            with open(f'{save_path}/log.txt', 'a') as file: 
                # file.write(f"iteraton:{ep*len(tar_finetuneloader)+index},FT_loss:{loss.item()}\n") 
                file.write(f"iteraton:{count},FT_loss:{loss.item()}\n") 
            loss.backward()
            log_grad_module(model,save_path,"when FT")

            optimizer.step()
            scheduler.step()
            print(f"count:{count}\n")
            if count%1000==0 or count ==0:
                # import pdb;pdb.set_trace()
                img_store_path_true=img_store_path+f"/iteration{count}"
                os.makedirs(img_store_path_true,exist_ok=True)
                draw_img(model,img_store_path_true,prompt_paths)
                try:
                    plot_FT_loss(save_path)
                except:
                    print("plot_FT_loss has error")
            count=count+1

    return model