import torch

state_dict = torch.load('../../code/imagenet-autoencoder-main/results_decoder/small_imagenet_porn/4_12_18_49_5/ckpts/loop2299_ori0.005_ft0.072_qloss0.00031.pt')['state_dict']

new_state_dict = {key: value for key, value in state_dict.items() if not key.startswith('module.decoder')}
for key in list(new_state_dict.keys()):
    if key.startswith('decoder'):
        new_key = 'module.' + key  
        new_state_dict[new_key] = new_state_dict.pop(key)

torch.save({'state_dict':new_state_dict}, '../../code/imagenet-autoencoder-main/results_decoder/small_imagenet_porn/4_12_18_49_5/ckpts/newloop2299_ori0.005_ft0.072_qloss0.00031.pt')

new_state_dict = {}
for key in list(state_dict.keys()):
    if key.startswith('module'):
        new_key = key[7:]  
        new_state_dict[new_key] = state_dict[key]