import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetv2(nn.Module):
    
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=True).features
        self.backbone1 = nn.Sequential(backbone[:3])
        self.backbone2 = nn.Sequential(backbone[3:])
        self.tr_conv = nn.ConvTranspose2d(1280, 1280, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn0 = nn.BatchNorm2d(24)
        self.bn1 = nn.BatchNorm2d(1280)
        self.act = nn.LeakyReLU()
        self.conv = nn.Conv2d(1280, 2048, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2048)
    
    def forward(self, x):

        out1 = self.backbone1(x)
        out1 = self.act(self.bn0(out1))
        out2 = self.backbone2(out1)
        out2 = self.act(self.bn1(self.tr_conv(out2)))
        out2 = self.act(self.bn2(self.conv(out2)))
        return out1, out2


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1
    ):
        super().__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size, 1, padding, dilation))
        self.add_module('bn', nn.BatchNorm3d(out_channels))
        self.add_module('act', nn.Mish())
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze().permute(1, 0, 2, 3)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return ((x * y.expand_as(x)).unsqueeze(0)).permute(0, 2, 1, 3, 4)
    
class ASSP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASSP, self).__init__()
        self.conv_1x1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            dilation=1
        )
        self.conv_6x6 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=6,
            dilation=6
        )
        self.conv_12x12 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=12,
            dilation=12
        )
        self.conv_18x18 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=18,
            dilation=18
        )
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, None, None)),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dilation=1
            )
        )
        self.final_conv = ConvBlock(
            in_channels=out_channels * 5,
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
        x5 = self.avg_pool(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.final_conv(x)
        return x
    
import torch.nn.functional as F

class DeepLabv3Plus(nn.Module):

    def __init__(self, num_classes=2):
        super(DeepLabv3Plus, self).__init__()
        self.backbone = MobileNetv2()
        self.se = SELayer(2048)
        self.assp = ASSP(in_channels=2048, out_channels=256)
        self.conv1x1_for_backbone = ConvBlock(
            in_channels = 24,
            out_channels=128,
            kernel_size = 1,
            padding=0,
            dilation=1
        )
        self.conv1x1_for_assp = ConvBlock(
            in_channels = 256,
            out_channels=128,
            kernel_size = 1,
            padding=0,
            dilation=1
        )
        self.conv3x3 = ConvBlock(
            in_channels = 256,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.classifier = ConvBlock(
            in_channels = 128,
            out_channels = num_classes,
            kernel_size = 1
        )

    def forward(self, x):

        out1, out = self.backbone(x.squeeze().permute(1, 0, 2, 3))
        out1 = out1.unsqueeze(0).permute(0, 2, 1, 3, 4)
        out = out.unsqueeze(0).permute(0, 2, 1, 3, 4)
        out = self.se(out) 
        out_assp = self.assp(out)
        out_assp = self.conv1x1_for_assp(out_assp)
        out_assp = F.interpolate(out_assp, scale_factor=(1, 4, 4))

        #out_back = nn.Sequential((self.backbone.backbone)[:3])(x.squeeze().permute(1, 0, 2, 3)).unsqueeze(0).permute(0, 2, 1, 3, 4)
        out_back = out1
        out_back = self.conv1x1_for_backbone(out_back)

        out = torch.cat([out_assp, out_back], dim=1)
        out = self.conv3x3(out)
        out = F.interpolate(out, scale_factor=(1, 4, 4))
        out = self.classifier(out)
        return out