import torch
import torch.nn as nn
from collections.abc import Sequence
from torchvision import models
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

#################

import torch
import torch.nn as nn
from collections.abc import Sequence
from torchvision import models
import torch.nn.functional as F

"""
input.shape = (BATCH_SIZE, CHANNELS, TIME, HEIGHT, WIDTH)
"""

def get_norm_layer(n_channels):
    return nn.BatchNorm3d(n_channels)

def get_act_layer():
    return nn.LeakyReLU(0, 1)

class ADN(nn.Sequential):
    """
    Обвёртка над слоями нормализации + активации.
    """
    def __init__(
        self,
        in_channels
    ):
        """
        in_channels - количество входных каналов, параметр С
        """
        super().__init__()
        self.add_module("N", get_norm_layer(in_channels))
        self.add_module("A", get_act_layer())
        
        
class Convolution(nn.Sequential):
    def __init__(
        self,
        in_channels, # количество входных каналов
        out_channels, # количество выходных каналов
        kernel_size,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size, 1, padding, dilation))
        self.add_module("adn", ADN(out_channels))
     
        
class ASSP(nn.Module):
    """
    Пирамидальная свёртка - encoder DeepLabv3+.
    """
    def __init__(self, in_channels, out_channels):
        
        super(ASSP, self).__init__()
        self.conv_1x1 = Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=1
        )
        
        self.conv_6x6 = Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=6,
            dilation=6
        )
        
        self.conv_12x12 = Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=12,
            dilation=12
        )
        
        self.conv_18x18 = Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=18,
            dilation=18
        )
        
        self.conv_24x24 = Convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=24,
            dilation=24
        )
        
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, None, None)),
            Convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dilation=1
            )
        )
        
        self.final_conv = Convolution(
            in_channels=out_channels * 6,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=1
        )
        
    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv_6x6(x)
        x3 = self.conv_12x12(x)
        x4 = self.conv_18x18(x)
        x5 = self.conv_24x24(x)
        x6 = self.image_pool(x)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = self.final_conv(x)
        return x
    

class Efficientnet_b7(nn.Module):
    def __init__(self, output_layer='features'):
        super(Efficientnet_b7, self).__init__()
        self.pretrained = models.efficientnet_b7(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None
        
    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
        return x
    
class ResNet101_deeplab(nn.Module):
    
    def __init__(self):
        super().__init__()
        resnet101 = models.resnet101(pretrained=True)
        backbone = torch.nn.Sequential(*list(resnet101.children())[:-2])
        backbone_output = backbone[:-1]
        self.backbone = backbone_output
        self.conv = nn.Conv2d(1024, 2048, 1)
        self.bn = nn.BatchNorm2d(2048)
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        with torch.no_grad():   
            x = self.backbone(x)
        x = self.conv(x)
        return self.act(self.bn(x))
    

class Deeplabv3Plus(nn.Module):
    
    '''
    input = [1, C, T, H, W] // first - 1 is equal that batch_size=1
    
    processing:
    - backbone: [1, C, H , W] -> (1, 2048, 32, 32)  
    '''
    
    def __init__(self, num_classes):
        super(Deeplabv3Plus, self).__init__()
        
        self.backbone = ResNet101_deeplab()
        
        self.assp = ASSP(in_channels = 2048, out_channels=256)
        
        self.conv1x1 = Convolution(
            in_channels = 256,
            out_channels=48,
            kernel_size = 1,
            padding=0,
            dilation=1
        )
        
        self.conv3x3 = Convolution(
            in_channels = 304,
            out_channels = 256,
            kernel_size = 3,
            padding = 1
        )
        
        self.classifier = Convolution(
            in_channels = 256,
            out_channels = num_classes,
            kernel_size = 1
        )
        
    def forward(self, x):
        
        # backbone processing
        x_backbone = None
        for i in range(x.shape[2]):
            cur_img = x[:,:,i, :, :]
            cur_img_backbone = self.backbone(cur_img).unsqueeze(2)
            if x_backbone is None:
                x_backbone = cur_img_backbone
            else:
                x_backbone = torch.cat([x_backbone, cur_img_backbone], dim=2)
        # assp processing 
        x_assp = self.assp(x_backbone)
        
        # interpolation
        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor = (1, 4, 4)
        )
        
        x_conv1x1 = self.conv1x1(x_assp_upsampled)
        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv3x3(x_cat)
        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor = (1, 8, 8)
        )
        x_out = self.classifier(x_3x3_upscaled)
        return x_out