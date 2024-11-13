import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock2D import Conv2D_block as ConvBlock2D

class DuckNet(nn.Module):
    
    def __init__(self,
                in_channels,
                out_channels,
                n_classes = 2
                ):
        super().__init__()
        
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2*out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=2*out_channels,
            out_channels=4*out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=4*out_channels,
            out_channels=8*out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.conv4 = nn.Conv2d(
            in_channels=8*out_channels,
            out_channels=16*out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.conv5 = nn.Conv2d(
            in_channels=16*out_channels,
            out_channels=32*out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.duck_block_1 = ConvBlock2D(in_channels=in_channels, out_channels=out_channels, block_type='duckv2')
        self.pool1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels*2,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.duck_block_2 = ConvBlock2D(in_channels=2*out_channels, out_channels=out_channels*2, block_type='duckv2')
        self.pool2 = nn.Conv2d(
            in_channels=out_channels*2,
            out_channels=out_channels*4,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.duck_block_3 = ConvBlock2D(in_channels=out_channels*4, out_channels=out_channels*4, block_type='duckv2')
        self.pool3 = nn.Conv2d(
            in_channels=out_channels*4,
            out_channels=out_channels*8,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.duck_block_4 = ConvBlock2D(in_channels=out_channels*8, out_channels=out_channels*8, block_type='duckv2')
        self.pool4 = nn.Conv2d(
            in_channels=out_channels*8,
            out_channels=out_channels*16,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.duck_block_5 = ConvBlock2D(in_channels=out_channels*16, out_channels=out_channels*16, block_type='duckv2')
        self.pool5 = nn.Conv2d(
            in_channels=out_channels*16,
            out_channels=out_channels*32,
            kernel_size=2,
            stride=2,
            padding=0
        )
        
        self.resnet1 = ConvBlock2D(in_channels=out_channels*32, out_channels=out_channels*32, block_type='resnet', repeat=2)
        self.resnet2 = ConvBlock2D(in_channels=out_channels*32, out_channels=out_channels*16, block_type='resnet', repeat=2)
        
        self.up_duck_block1 = ConvBlock2D(in_channels=16*out_channels, out_channels=8*out_channels, block_type='duckv2')
        self.up_duck_block2 = ConvBlock2D(in_channels=8*out_channels, out_channels=4*out_channels, block_type='duckv2')
        self.up_duck_block3 = ConvBlock2D(in_channels=4*out_channels, out_channels=2*out_channels, block_type='duckv2')
        self.up_duck_block4 = ConvBlock2D(in_channels=2*out_channels, out_channels=out_channels, block_type='duckv2')
        self.up_duck_block5 = ConvBlock2D(in_channels=out_channels, out_channels=out_channels, block_type='duckv2')
        
        self.final_conv = nn.Conv2d(in_channels=out_channels, out_channels=n_classes, kernel_size=1)
        
    def forward(self, input):
        
        p1 = self.conv1(input)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)
        
        t0 = self.duck_block_1(input)
        l1i = self.pool1(t0)
        s1 = l1i + p1
        
        t1 = self.duck_block_2(s1)
        l2i = self.pool2(t1)
        s2 = l2i + p2
        
        t2 = self.duck_block_3(s2)
        l3i = self.pool3(t2)
        s3 = l3i + p3
        
        t3 = self.duck_block_4(s3)
        l4i = self.pool4(t3)
        s4 = l4i + p4
        
        t4 = self.duck_block_5(s4)
        l5i = self.pool5(t4)
        s5 = l5i + p5
        
        t51 = self.resnet1(s5)
        t53 = self.resnet2(t51)
        
        l5o = F.interpolate(t53, scale_factor=2)
        c4 = l5o + t4
        q4 = self.up_duck_block1(c4)
        
        l4o = F.interpolate(q4, scale_factor=2)
        c3 = l4o + t3
        q3 = self.up_duck_block2(c3)
        
        l3o = F.interpolate(q3, scale_factor=2)
        c2 = l3o + t2
        q6 = self.up_duck_block3(c2)
        
        l2o = F.interpolate(q6, scale_factor=2)
        c1 = l2o + t1
        q1 = self.up_duck_block4(c1)
        
        l1o = F.interpolate(q1, scale_factor=2)
        c0  = l1o + t0
        z1 = self.up_duck_block5(c0)
        
        output = self.final_conv(z1)
        
        return output
        
        

        
        
        