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

def get_norm_layer(norm, spatial_dims, channels):
    
    '''
    Возвращает слой батч-нормализации для 3d свёртки.
    '''

    return nn.BatchNorm3d(channels)

def get_act_layer(act):

    '''
    Возвращает функцию активации.
    '''

    return nn.LeakyReLU(0.1)

class ADN(nn.Sequential):
    
    '''
    Обвёртка над слоями нормализация + активация.
    '''

    def __init__(self,
        in_channels: int | None = None,
        act: tuple | str | None = 'RELU',
        norm: tuple | str | None = None,
        norm_dim: int | None = None,
        dropout_dim: int | None = None,
    ) -> None:
        super().__init__()
        op_dict = {'N' : None, 'A' : None}
        if norm is not None:
            if norm_dim is None:
                raise ValueError('norm_dim needs to be specified.')
            op_dict['N'] = get_norm_layer(norm=norm, spatial_dims=norm_dim, channels=in_channels)
            
        if act is not None:
            op_dict['A'] = get_act_layer(act)
            
        self.add_module('N', op_dict['N'])
        self.add_module('A', op_dict['A'])

class Convolution(nn.Sequential):
    
    """
    Constructs a 3d-convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --
    """
    
    def __init__(
      self,
        in_channels: int,
        out_channels: int ,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        norm: tuple | str | None = 'INSTANCE',
        dropout: tuple  | str | float | None = None,
        bias: bool = True,
        conv_only: bool = False,
        padding: Sequence[int] | int | None = 0,
        dilation: Sequence[int] | int = 1,
        output_padding : Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = strides,
            padding = padding,
            dilation = dilation,
            bias = bias,
        )
        self.add_module('conv', conv)
        
        if conv_only:
            return
        self.add_module('adn',
                    ADN(
                        in_channels = out_channels,
                        act='RELU',
                        norm = norm,
                        norm_dim=3,
                    ))

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
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, None, None)),
            Convolution(
                in_channels = in_channels,
                out_channels=out_channels,
                kernel_size = 1,
                strides=1,
                padding=0,
                dilation=1
            )
        )
      
        self.final_conv = Convolution(
            in_channels = 5 * out_channels,
            out_channels = out_channels,
            kernel_size=1,
            padding=0,
            dilation=1
        )
        
    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)
        img_pool_opt = self.image_pool(x)
        
        concat = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt),
            dim=1
        )
        x_final_conv = self.final_conv(concat)
        return x_final_conv

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
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None
    
    def forward(self, x):
        with torch.no_grad():
            x = self.net(x)
        return x

class Deeplabv3Plus(nn.Module):
    
    '''
    input = [1, C, T, H, W] // first - 1 is equal that batch_size=1
    
    processing:
    - backbone: [1, C, H , W] -> (1, 2560, 16, 16)  
    '''
    
    def __init__(self, num_classes):
        
        super(Deeplabv3Plus, self).__init__()
        
        self.backbone = Efficientnet_b7()
        
        self.assp = ASSP(in_channels = 2560, out_channels=256)
        
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