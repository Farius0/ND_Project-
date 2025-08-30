import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from .GLCA import GLCA
from .Fusion import Fusion

class ConvBlock(nn.Module):
    """two convolution labels with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            # ConvBlock(out_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]
    
class FEUNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(FEUNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.0, 0.0, 0.0, 0.0, 0.0], # [0.05, 0.1, 0.2, 0.3, 0.5], [0.01, 0.025, 0.05, 0.075, 0.1]
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']

        self.encoder = Encoder(params)

        self.glca4 = GLCA(dim_in=params["feature_chns"][3])
        self.glca3 = GLCA(dim_in=params["feature_chns"][2])
        self.glca2 = GLCA(dim_in=params["feature_chns"][1])
        self.glca1 = GLCA(dim_in=params["feature_chns"][0])

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.fusion1 = Fusion(in_channels=params["feature_chns"][3], out_channels=params["feature_chns"][2])
        self.fusion2 = Fusion(in_channels=params["feature_chns"][2], out_channels=params["feature_chns"][1])
        self.fusion3 = Fusion(in_channels=params["feature_chns"][1], out_channels=params["feature_chns"][0])

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(self.ft_chns[0], self.n_class - 1, kernel_size=3, padding=1)


    def forward(self, x):
        
        h0, w0 = x.shape[-2:]  # Padding if needed
        pad_h = (16 - h0 % 16) % 16
        pad_w = (16 - w0 % 16) % 16
        
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        feature = self.encoder(x)

        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        features = feature

        g4 = self.glca4(x3)
        g3 = self.glca3(x2)
        g2 = self.glca2(x1)

        d4 = self.up1(x4, g4)
        d3 = self.up2(d4, g3)
        d2 = self.up3(d3, g2)
        d1 = self.up4(d2, x0)

        # MSF
        fusion43 = self.fusion1(d4, d3)
        fusion432 = self.fusion2(fusion43, d2)
        fusion4321 = self.fusion3(fusion432, d1)

        output = self.out_conv(fusion4321)
        reg_out = self.reg_conv(fusion4321)
        
        if pad_h > 0 or pad_w > 0:
            output = output[..., :h0, :w0]
            reg_out = reg_out[..., :h0, :w0]

        return {"features": features, "seg_out": output, "reg_out": reg_out}

