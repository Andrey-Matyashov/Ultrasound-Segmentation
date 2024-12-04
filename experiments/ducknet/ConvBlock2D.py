import torch
import torch.nn as nn


class SeparatedConv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels,
                size=3, padding="same"):
        
        """
        in_channels: number of input channels
        out_channels: number of output channels
        """
        
        super(SeparatedConv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(size, 1),
            padding=((size - 1) // 2, 0),
        )
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, size),
            padding=(0, (size - 1) // 2),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        return x
    
class MidscopeConv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        """
        in_channels: number of input channels
        out_channels: number of output channels
        """
        
        super(MidscopeConv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            
        )
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        return x
    
    
class WidescopeConv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        """
        in_channels: number of input channels
        out_channels: number of output channels
        """
        
        super(WidescopeConv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=2,
        )
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=2,
            dilation=3
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.bn3(x)
        return x
    
    
class ResNetConv2D_block(nn.Module):
    """
    При использовании данного класса параметр dilation должен быть равен 1
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResNetConv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=dilation
        )
        
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.bn2(x)
        x_final = x1 + x
        return self.bn3(x_final)
    
class DoubleConvAndBatchNorm(nn.Module):
    
    def __init__(self, in_channels, out_channels, dilation=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation
        )
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)
        return x
    
class Conv2D_block(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 block_type,
                 repeat=1,
                 dilation=1,
                 size=3,    
            ):
        super(Conv2D_block, self).__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(repeat): 
            if i > 0:
                in_channels = out_channels
            if block_type == 'separated':
                self.blocks.append(SeparatedConv2D_block(in_channels=in_channels, out_channels=out_channels, size=size))
            elif block_type == 'duckv2':
                self.blocks.append(DuckConv2D_block(in_channels=in_channels, out_channels=out_channels, size=size))
            elif block_type == 'midscope':
                self.blocks.append(MidscopeConv2D_block(in_channels=in_channels, out_channels=out_channels))
            elif block_type == 'widescope':
                self.blocks.append(WidescopeConv2D_block(in_channels=in_channels, out_channels=out_channels))
            elif block_type == 'resnet':
                self.blocks.append(ResNetConv2D_block(in_channels=in_channels, out_channels=out_channels, dilation=dilation))
            elif block_type == 'double_convolution':
                self.blocks.append(DoubleConvAndBatchNorm(in_channels=in_channels, out_channels=out_channels, dilation=dilation))
            elif block_type == 'conv':
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=size,
                    padding=1,
                    dilation=dilation
                ),
                nn.LeakyReLU(),
            ))
            else:
                raise ValueError(f'Unknown block type: {block_type}')
        
        self.repeat = repeat
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        
    
    
class DuckConv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(DuckConv2D_block, self).__init__()
        self.widescope = WidescopeConv2D_block(in_channels=in_channels, out_channels=out_channels)
        self.midscope = MidscopeConv2D_block(in_channels=in_channels, out_channels=out_channels)
        self.resnet1 = Conv2D_block(in_channels=in_channels, out_channels=out_channels, block_type='resnet', dilation=1, size=size)
        self.resnet2 = Conv2D_block(in_channels=in_channels, out_channels=out_channels, block_type='resnet', dilation=1, size=size, repeat=2)
        self.resnet3 = Conv2D_block(in_channels=in_channels, out_channels=out_channels, block_type='resnet', dilation=1, size=size, repeat=3)
        self.separated = SeparatedConv2D_block(in_channels=in_channels, out_channels=out_channels, size=size)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        x = self.bn1(x)
        
        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x)
        x5 = self.resnet3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn2(x)
        return x