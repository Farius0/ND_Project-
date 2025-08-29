# ==================================================
# ==============  MODULE: FEUNet_v2 ================
# ==================================================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- config ---
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "feunet_config.yaml")

with open(config_path, "r") as f:
    default_config = yaml.safe_load(f)
    
# ------------------------- basic blocks -------------------------    

class ConvBlock(nn.Module):
    """
    Two Conv-BatchNorm-LeakyReLU layers with optional dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_conv(x)
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """
    Downsampling followed by a ConvBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UpBlock(nn.Module):
    """
    Upsample (bilinear or transposed) + skip connection + ConvBlock.
    """

    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int, dropout_p: float = 0.0, bilinear: bool = True) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels1, in_channels2, kernel_size=1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2, bias=False)

        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class DilatedConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

# ------------------------- GLCA and attention blocks -------------------------

class GLCA(nn.Module):
    """
    Global And Local Context-aware
    """
    def __init__(self, dim_in: int) -> None:
        super(GLCA, self).__init__()
        self.in_channels = dim_in
        self.inter_channels = 32 # if dim_in < 32 else dim_in
        
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
        
        modules = []
        rates = (1, 3, 5)
        for rate in rates:
            modules.append(DilatedConv(self.inter_channels, self.inter_channels, rate))
        self.convs = nn.ModuleList(modules)

    def forward(self, feature):

        B, C, H, W = feature.size()

        # (B*W, H, Cint) @ (B*W, Cint, H) -> (B*W, H, H)
        query = self.query(feature).permute(0, 3, 2, 1).contiguous().view(B * W, H, self.inter_channels)
        key = self.key(feature).permute(0, 3, 1, 2).contiguous().view(B * W, self.inter_channels, H)
        att_map = self.softmax(torch.matmul(query, key))  # (B*W, H, H)
        
        value = self.value(feature)  # (B, Cint, H, W)

        outs = []
        for conv in self.convs:
            v = conv(value).permute(0, 3, 2, 1).contiguous().view(B * W, H, self.inter_channels)  # (B*W, H, Cint)
            outs.append(torch.matmul(att_map, v))  # (B*W, H, Cint)

        res = torch.cat(outs, dim=2)  # (B*W, H, Cint * len(convs))
        res = res.view(B, W, H, -1).permute(0, 3, 2, 1)  # (B, Cint*len, H, W)

        out = self.prelu(self.conv1x1(res) + feature)
        return out

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention mechanism.
    """

    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        mid = max(1, in_channels // ratio)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)

# ---------------------------------- Fusion block -------------------------

class Fusion(nn.Module):
    """
    Feature fusion with attention-weighted blending.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attn = ChannelAttention(out_channels * 2)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, kernel_size=3, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_l: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
        x_l = self.upsample(x_l)
        x_l = self.project(x_l)
        x_cat = torch.cat([x_l, x_h], dim=1)
        attention = self.attn(x_cat)
        weighted = self.fusion_conv(x_cat * attention)
        weights = self.sigmoid(weighted)
        return (x_l + x_h) * weights
    

# ---------------------------------- Encoder -------------------------

class Encoder(nn.Module):
    """
    UNet-style encoder that dynamically adapts to any number of levels.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    feature_channels : list of int
        List of output channels for each encoder level.
        Example: [16, 32, 64, 128] means 4 levels.
    dropouts : list of float
        Dropout rates corresponding to each encoder level.
        Must be the same length as feature_channels.
    """

    def __init__(self, in_channels: int, feature_channels: List[int], dropouts: List[float]) -> None:
        super().__init__()
        assert len(feature_channels) == len(dropouts), "feature_channels and dropouts must be the same length."

        # Initial convolution block
        self.in_conv = ConvBlock(in_channels, feature_channels[0], dropouts[0])

        # Downsampling layers
        self.down_blocks = nn.ModuleList([
            DownBlock(feature_channels[i - 1], feature_channels[i], dropouts[i])
            for i in range(1, len(feature_channels))
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        list of torch.Tensor
            Feature maps from each encoder level, including the input conv layer.
        """
        feats: List[torch.Tensor] = []
        x = self.in_conv(x); feats.append(x)
        for block in self.down_blocks:
            x = block(x); feats.append(x)
        return feats


# ---------------------------------- Full FEUNet_v2 Model -------------------------

class FEUNet_v2(nn.Module):
    def __init__(self, config: dict = default_config):
        super().__init__()

    def __init__(self, config: Dict = default_config) -> None:
        super().__init__()
        self.config = config
        self.in_channels: int = config["in_channels"]
        self.out_channels: int = config["out_channels"]
        self.feature_channels: List[int] = config["feature_channels"]
        self.dropouts: List[float] = config["dropouts"]
        self.use_glca: bool = config.get("use_glca", False)
        self.use_fusion: bool = config.get("use_fusion", False)
        self.bilinear: bool = config.get("bilinear", False)
        default_levels = list(range(1, len(self.feature_channels) - 1))
        self.glca_levels: List[int] = config.get("glca_levels", default_levels)
        self.debug_features: Dict[str, torch.Tensor] = {}  # Store hook outputs here

        # Encoder
        self.encoder = Encoder(
            in_channels=self.in_channels,
            feature_channels=self.feature_channels,
            dropouts=self.dropouts
        )

        # GLCA blocks (if used)
        if self.use_glca:
            self.glcas = nn.ModuleList([
                GLCA(dim_in=self.feature_channels[i])
                for i in range(len(self.feature_channels))
                if i in self.glca_levels
            ])
            self.glca_mapping = {i: idx for idx, i in enumerate(sorted(self.glca_levels))}
        else:
            self.glcas = None
            self.glca_mapping = {}
            
        # Decoder (UpBlocks)            
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_channels1=self.feature_channels[i + 1],
                in_channels2=self.feature_channels[i],
                out_channels=self.feature_channels[i],
                dropout_p=self.dropouts[i],
                bilinear=self.bilinear
            )
            for i in reversed(range(len(self.feature_channels) - 1))
        ])
        
        # Fusion blocks (if used)

        if self.use_fusion:
            self.fusion_blocks = nn.ModuleList([
                Fusion(
                    in_channels=self.feature_channels[i + 1],
                    out_channels=self.feature_channels[i]
                )
                for i in reversed(range(len(self.feature_channels) - 2))
            ])
            
        else:
            self.fusion_blocks = None

        # Final conv layers for segmentation and regularization outputs
        self.out_conv = nn.Conv2d(self.feature_channels[0], self.out_channels, kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(self.feature_channels[0], max(1, self.out_channels - 1), kernel_size=3, padding=1) \
                        if self.out_channels > 1 else None

        # # Initialize weights
        # self._initialize_weights()
        
        # # Encoder levels
        # for i, block in enumerate(self.encoder.down_blocks):
        #     self._register_hook(block, f"encoder_down{i+1}")

        # # GLCA
        # if self.use_glca:
        #     for i, glca in enumerate(self.glcas):
        #         self._register_hook(glca, f"glca{i+1}")

        # # UpBlocks
        # for i, up in enumerate(self.up_blocks):
        #     self._register_hook(up, f"upblock{i+1}")

        # # Fusion blocks
        # if self.use_fusion:
        #     for i, fusion in enumerate(self.fusion_blocks):
        #         self._register_hook(fusion, f"fusion{i+1}")


    def _initialize_weights(self) -> None:
        """
        Apply Kaiming normal initialization to conv layers,
        and standard initialization to batch norms.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                    
    def _register_hook(self, module: nn.Module, name: str, store: bool = True, verbose: bool = False) -> None:
        def hook_fn(module, input, output):
            if store:
                self.debug_features[name] = output.detach()
            if verbose:
                print(f"{name} ({module.__class__.__name__}) → {tuple(output.shape)}")
        module.register_forward_hook(hook_fn)
      
    def clear_debug_features(self) -> None:
        self.debug_features = {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Forward pass through the full network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        dict
            Dictionary with segmentation and regularization outputs.
        """
        
        x, pad_h, pad_w = pad_to_multiple(x)
        
        features = self.encoder(x)

        # Apply GLCA (if used)
        if self.use_glca:
            for i in self.glca_levels:
                features[i] = self.glcas[self.glca_mapping[i]](features[i])

        # Decoder
        x = features[-1]
        up_outputs: List[torch.Tensor] = []

        for i, up in enumerate(self.up_blocks):
            skip_feat = features[-(i + 2)]
            x = up(x, skip_feat)
            up_outputs.append(x)

        # Fusion blocks (if used)
        if self.use_fusion:
            x = up_outputs[0]
            for i, fusion in enumerate(self.fusion_blocks):
                x = fusion(x, up_outputs[i + 1])

        # Final outputs
        seg_out = self.out_conv(x)
        reg_out = self.reg_conv(x) if self.reg_conv is not None else None
        
        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            seg_out = crop_to_original(seg_out, pad_h, pad_w)
            if reg_out is not None:
                reg_out = crop_to_original(reg_out, pad_h, pad_w)

        out: Dict[str, torch.Tensor | List[torch.Tensor]] = {"features": features, "seg_out": seg_out}
        if reg_out is not None:
            out["reg_out"] = reg_out
        return out
    
# ------------------------- padding utils -------------------------

def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, int, int]:

    """
    Pad bottom and right of a tensor to make height and width multiples of `multiple`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W)
    multiple : int
        Target multiple for H and W.

    Returns
    -------
    x_padded : torch.Tensor
        Tensor padded on bottom and right.
    pad_h : int
        Number of pixels added to height.
    pad_w : int
        Number of pixels added to width.
    """
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # pad (left, right, top, bottom)
    return x_padded, pad_h, pad_w


def crop_to_original(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """
    Crop tensor back to original size by removing padding from bottom and right.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (B, C, H + pad_h, W + pad_w)
    pad_h : int
        Pixels to remove from bottom.
    pad_w : int
        Pixels to remove from right.

    Returns
    -------
    x_cropped : torch.Tensor
        Tensor of original shape (B, C, H, W)
    """
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :, :-pad_w]
    return x