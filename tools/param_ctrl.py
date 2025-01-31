import torch.nn as nn
import torch 
def init_weights(m):
    if isinstance(m, nn.Linear):
        
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
       
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
       
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
def unet_and_lora_require_grad(model,enable,targets=["diffusion_model"],init=False):
    if targets !=None:
        for name, module in model.named_parameters():
            # if targets[0]== "diffusion_model":
                for target in targets:
                    
                    if (target in name):
                        # import pdb;pdb.set_trace()
                        if enable:
                            if module.requires_grad != True:
                                module.requires_grad = True
                        else:
                            # module.detach()
                            # import pdb;pdb.set_trace()
                            if module.requires_grad != False:
                                module.requires_grad = False
                        
                        break
                    
                    else:
                        # module.detach()
                        # import pdb;pdb.set_trace()
                        if module.requires_grad != False:
                            module.requires_grad = False
    if init==True:
        for target in targets:
            if target == "diffusion_model":
                model.model.diffusion_model.apply(init_weights)
            else:
                raise ValueError("only diffusion_model can be init for now")

def unet_and_lora_require_grad_detach(model,enable,detach,targets=["diffusion_model"],init=False):
    if targets !=None:
        for name, module in model.named_parameters():
            # if targets[0]== "diffusion_model":
                for target in targets:
                    
                    if (target in name):
                        # import pdb;pdb.set_trace()
                        if enable:
                            if module.requires_grad != True:
                                module.requires_grad = True
                        else:
                            # module.detach()
                            # import pdb;pdb.set_trace()
                            module.requires_grad = False
                        
                        break
                    
                    else:
                        # if detach==True:
                        #     module.detach()
                        #     print("detach \n")
                        module.requires_grad = False
    if init==True:
        for target in targets:
            if target == "diffusion_model":
                model.model.diffusion_model.apply(init_weights)
            else:
                raise ValueError("only diffusion_model can be init for now")
def log_grad_module(model,save_path,fname):
    with open(f"{save_path}/{fname}.txt", "w") as f:
        for name, param in model.named_parameters():
            # if 'diffusion_model' in name:
            f.write(f"Layer: {name} | requires_grad: {param.requires_grad}| is leaf :{param.is_leaf}\n")
            # elif param.requires_grad ==True:
                # f.write(f"*******Layer: {name} | requires_grad: {param.requires_grad}******\n")


