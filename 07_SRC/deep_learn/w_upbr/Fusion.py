import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttetion(nn.Module):

    """
    SE Channel Attention
    """

    def __init__(self, in_channels, ratio):
        super(ChannelAttetion, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0)
        self.excitation = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return torch.sigmoid(out)

class Fusion(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttetion(in_channels=out_channels * 2, ratio=16)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels // 2, out_channels=1, kernel_size=3, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_l, x_h):
        x_l = self.up_sample(x_l)
        x_l = self.conv1x1(x_l)
        x_c = torch.cat([x_l, x_h], dim=1)
        x_add = x_l + x_h
        x_ca = x_c * self.ca(x_c)
        x_conv = self.conv(x_ca)
        x = self.softmax(x_conv)
        out = x_add * x

        return out


