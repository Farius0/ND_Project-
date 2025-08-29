import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class GLCA(nn.Module):
    """
    Global And Local Context-aware
    """
    def __init__(self, dim_in):
        super(GLCA, self).__init__()

        self.in_channels = dim_in
        self.inter_channels = 32 #if dim_in < 32 else dim_in
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=self.inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.PReLU()
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=self.inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.PReLU()
        )
        self.value = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=self.inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.PReLU()
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels * 3, out_channels=dim_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_in),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.prelu = nn.PReLU()
        atrous_rates = [1, 3, 5]
        modules = []
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(DilatedConv(self.inter_channels, self.inter_channels, rate))
        self.convs = nn.ModuleList(modules)

    def forward(self, feature):

        B, C, H, W = feature.size()

        query = self.query(feature).permute(0, 3, 2, 1)
        query = query.contiguous().view(B * W, H, self.inter_channels)
        key = self.key(feature).permute(0, 3, 1, 2)
        key = key.contiguous().view(B * W, self.inter_channels, H)
        att_map = torch.matmul(query, key)
        att_map = self.softmax(att_map)
        value = self.value(feature)

        _res = []
        for conv in self.convs:
            temp = conv(value).permute(0, 3, 1, 2)
            _value = temp.contiguous().view(B * W, H, self.inter_channels)
            _res.append(torch.matmul(att_map, _value))

        res = torch.cat(_res, dim=0)
        res = res.view(B, W, H, -1).permute(0, 3, 2, 1)
        out = self.prelu(self.conv1x1(res) + feature)
        return out
























