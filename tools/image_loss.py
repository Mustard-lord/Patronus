
import torch
from pytorch_msssim import ssim
import sys
import torch.nn as nn
from typing import cast, Dict, List, Union
sys.path.append('../')
from collections import namedtuple
import torch.nn.functional as F
from pytorch_msssim import ssim
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def _vgg(cfg: str, batch_norm: bool, weights):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    return model

def initial():
    model = _vgg("E", False, "../../.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth")
    return model

LossOutput = namedtuple("LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

def vggloss(img1, img2):
    vgg_model = initial()
    vgg_model = vgg_model.eval()
    loss_network = LossNetwork(vgg_model).cuda()
    loss_network = torch.nn.DataParallel(loss_network)
    features_full = loss_network(img1)
    features_input = loss_network(img2)
    content_loss = F.mse_loss(features_full[2], features_input[2])
    return content_loss

if __name__ == '__main__':
    print(vggloss(torch.zeros(5,3,32,32).cuda(), torch.randn(5,3,32,32).cuda()))
    print(vggloss(torch.zeros(100,3,32,32).cuda(), torch.randn(100,3,32,32).cuda()))
    
    # psnr(torch.randn(3,256,256).cuda(), torch.randn(3,256,256).cuda())
    # print(ssim(torch.randn(10,3,256,256).cuda(), torch.randn(10,3,256,256).cuda()))
    
   