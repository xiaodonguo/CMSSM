import torch
import torch.nn as nn
from torch import Tensor
from backbone.efficientvit.models.nn.act import build_act
from backbone.efficientvit.models.nn.norm import build_norm
from backbone.efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple
from torch.cuda.amp import autocast
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from model_others.RGB_T.CMX.models.net_utils import FeatureFusionModule as FFM
from model_others.RGB_T.CMX.models.net_utils import FeatureRectifyModule as FRM
from timm.models.layers import trunc_normal_
import math
from model_others.RGB_T.MAINet import TSFA
from model_others.RGB_T.CLNet_T import MHFI
from proposed.fuison_strategy.MHFI_change import MHFI_change
# MD
from model_others.RGB_T.MDNet.model import MultiSpectralAttentionLayer, SS_Conv_SSM
from backbone.MedMamba import SS2D
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class Fusion_Module(nn.Module):
    def __init__(self, fusion_mode, channels, num_heads, scales=(5, ), depths=[2, 2, 4, 2], norm_fuse=nn.BatchNorm2d):
        super(Fusion_Module, self).__init__()
        self.fusion_mode = fusion_mode
        self.norm_layer = nn.ModuleList()
        self.fusion = nn.ModuleList()
        if self.fusion_mode == 'add':
            for i in channels:
                self.norm_layer.append(nn.BatchNorm2d(i))
        if self.fusion_mode == 'cat':
            for i in channels:
                # self.fusion.append(BasicConv(2 * i, i, 1, 1, 0))
                self.fusion.append(BasicConv(2 * i, i, 3, 1, 1))
                # self.fusion.append(BasicConv(2 * i, i, 5, 1, 2))
                # self.fusion.append(BasicConv(2 * i, i, 7, 1, 3))
                # self.fusion.append(BasicConv(2 * i, i, 9, 1, 4))
                # self.fusion.append(BasicConv(2 * i, i, 11, 1, 5))
                # self.fusion.append(BasicConv(2 * i, i, 13, 1, 6))
                # self.fusion.append(BasicConv(2 * i, i, 15, 1, 7))
        if self.fusion_mode == 'base':
            for i in channels:
                self.fusion.append(Base(2 * i, i))
        if self.fusion_mode == 'res':
            self.conv = nn.ModuleList([
                ResBlock(channels[0]*2),
                ResBlock(channels[1]*2),
                ResBlock(channels[2]*2),
                ResBlock(channels[3]*2)
            ])
        if self.fusion_mode == 'CMX':
            self.FRMs = nn.ModuleList([
                FRM(dim=channels[0], reduction=1),
                FRM(dim=channels[1], reduction=1),
                FRM(dim=channels[2], reduction=1),
                FRM(dim=channels[3], reduction=1)])

            self.FFMs = nn.ModuleList([
                FFM(dim=channels[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
                FFM(dim=channels[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
                FFM(dim=channels[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
                FFM(dim=channels[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])
        if self.fusion_mode == 'new':
            for channel in channels:
                self.fusion.append(NewFusion(channel, scales))

        if self.fusion_mode == 'scale':
            for channel in channels:
                self.fusion.append(MultiscalseLayer(channel * 2, channel, scales))
        if self.fusion_mode == 'dilation':
            for channel in channels:
                self.fusion.append(Dilation(channel * 2))

        if self.fusion_mode == 'TSFA':
            for channel in channels:
                self.fusion.append(TSFA(channel))

        if self.fusion_mode == 'MHFI':
            for channel in channels:
                self.fusion.append(MHFI(channel))

        if self.fusion_mode == 'MHFI_change':
            for i in range(4):
                if i == 0:
                    self.fusion.append(MHFI_change(channels[i], None, 1))
                else:
                    self.fusion.append(MHFI_change(channels[i], channels[i-1], i+1))

        if self.fusion_mode == 'MDFusion':
            self.fusion = MDFusion(channels)

        if self.fusion_mode == 'Mamba':
            for channel in channels:
                self.fusion.append(Mamba(channel))

        if self.fusion_mode == 'Mamba1':
            for channel in channels:
                self.fusion.append(Mamba1(channel))

        if self.fusion_mode == 'Mamba2':
            for channel, depth in zip(channels, depths):
                self.fusion.append(Mamba2(channel, depth))

        if self.fusion_mode == 'Mamba3':
            for channel, depth in zip(channels, depths):
                self.fusion.append(Mamba3(channel, depth))

        if self.fusion_mode == 'Mamba4':
            for channel, depth in zip(channels, depths):
                self.fusion.append(Mamba4(channel, depth))

        if self.fusion_mode == 'Mamba5':
            for channel, depth in zip(channels, depths):
                self.fusion.append(Mamba5(channel, depth))
    def forward(self, rgb, t):
        outs = []
        if self.fusion_mode == 'MDFusion':
            outs = self.fusion(rgb, t)
        for i in range(4):
            if self.fusion_mode == 'add':
                outs.append(self.norm_layer[i](rgb[i] + t[i]))
            if self.fusion_mode == 'max':
                out, _ = torch.max(torch.stack([rgb[i], t[i]], dim=1), dim=1)
                outs.append(self.norm_layer[i](out))
            if self.fusion_mode == 'cat':
                outs.append(self.fusion[i](torch.cat((rgb[i], t[i]), dim=1)))
            if self.fusion_mode == 'base':
                outs.append(self.fusion[i](rgb[i], t[i]))
            if self.fusion_mode == 'CMX':
                rgb_, t_ = self.FRMs[i](rgb[i], t[i])
                # rgb_, t_ = rgb[i], t[i]
                out = self.FFMs[i](rgb_, t_)
                outs.append(out)
            if self.fusion_mode == 'new':
                outs.append(self.fusion[i](rgb[i], t[i]))
            if self.fusion_mode == 'res':
                outs.append(self.conv[i](torch.cat((rgb[i], t[i]), dim=1)))
            if self.fusion_mode == 'scale':
                outs.append(self.fusion[i](rgb[i], t[i]))
            if self.fusion_mode == 'dilation':
                outs.append(self.fusion[i](rgb[i], t[i]))
            if self.fusion_mode == 'TSFA':
                outs.append(self.fusion[i](rgb[i], t[i]))
            if self.fusion_mode == 'MHFI':
                if i == 0:
                    outs.append(self.fusion[i](rgb[i], t[i], NO=i+1))
                else:
                    outs.append(self.fusion[i](rgb[i], t[i], former=outs[i-1], NO=i+1))
            if self.fusion_mode == 'MHFI_change':
                if i == 0:
                    outs.append(self.fusion[i](rgb[i], t[i]))
                else:
                    outs.append(self.fusion[i](rgb[i], t[i], former=outs[i-1]))
            if self.fusion_mode == 'Mamba':
                outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode in ['Mamba1', 'Mamba2', 'Mamba3', 'Mamba4', 'Mamba5']:
                outs.append(self.fusion[i](rgb[i], t[i]))

        return outs

class NewFusion(nn.Module):
    def __init__(self, in_channels, scales=(5,)):
        super(NewFusion, self).__init__()
        self.in_channels = in_channels
        self.MBConv1 = MBConv(
            in_channels=2*in_channels,
            out_channels=in_channels,
            stride=1,
            expand_ratio=0.5,
            act_func=("hswish", "hswish", None))
        self.fusion = LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=1,
                scales=scales,
                dim=32,
                norm=(None, "hswish"),
            )
        self.MBConv2 = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=4,
            use_bias=(True, True, False),
            norm=(None, None, "bn2d"),
            act_func=("hswish", "hswish", None),
        )
    def forward(self, rgb, t):
        B, C, H, W = rgb.size()
        rgbt = self.MBConv1(torch.cat((rgb, t), dim=1))
        # rgbt = torch.cat((rgb, t), dim=1)
        out = self.fusion(rgbt, rgb, t)
        out = self.MBConv2(out)
        return out

class Base(nn.Module):
    def __init__(self, in_c, out_c):
        super(Base, self).__init__()
        self.conv = BasicConv(in_c, out_c, 3, 1, 1)
    def forward(self, rgb, t):
        out = self.conv(torch.cat((rgb*t, rgb+t), dim=1))
        return out

class MDFusion(nn.Module):
    def __init__(self, channels):
        super(MDFusion, self).__init__()

        self.mca1 = SS_Conv_SSM(channels[0] // 4)
        self.mca2 = SS_Conv_SSM(channels[1] // 4)
        self.mca3 = SS_Conv_SSM(channels[2] // 4)
        self.mca4 = SS_Conv_SSM(channels[3] // 4)

        # self.dct1 = MultiSpectralAttentionLayer(channels[0] // 4, shape[0] // 4, shape[1] // 4)
        # self.dct2 = MultiSpectralAttentionLayer(channels[1] // 4, shape[0] // 8, shape[0] // 8)
        # self.dct3 = MultiSpectralAttentionLayer(channels[2] // 4, shape[0] // 16, shape[0] // 16)
        # self.dct4 = MultiSpectralAttentionLayer(channels[3] // 4, shape[0] // 32, shape[0] // 32)

        self.mca1t = SS_Conv_SSM(channels[0] // 4)
        self.mca2t = SS_Conv_SSM(channels[1] // 4)
        self.mca3t = SS_Conv_SSM(channels[2] // 4)
        self.mca4t = SS_Conv_SSM(channels[3] // 4)

        self.lpr1 = nn.Conv2d(channels[0], channels[0]//4, 1, 1, 0)
        self.lpr2 = nn.Conv2d(channels[1], channels[1]//4, 1, 1, 0)
        self.lpr3 = nn.Conv2d(channels[2], channels[2]//4, 1, 1, 0)
        self.lpr4 = nn.Conv2d(channels[3], channels[3]//4, 1, 1, 0)

        self.lpt1 = nn.Conv2d(channels[0], channels[0]//4, 1, 1, 0)
        self.lpt2 = nn.Conv2d(channels[1], channels[1]//4, 1, 1, 0)
        self.lpt3 = nn.Conv2d(channels[2], channels[2]//4, 1, 1, 0)
        self.lpt4 = nn.Conv2d(channels[3], channels[3]//4, 1, 1, 0)

        self.lp1 = nn.Conv2d(channels[0] // 2, channels[0]//4, 1, 1, 0)
        self.lp2 = nn.Conv2d(channels[1] // 2, channels[1]//4, 1, 1, 0)
        self.lp3 = nn.Conv2d(channels[2] // 2, channels[2]//4, 1, 1, 0)
        self.lp4 = nn.Conv2d(channels[3] // 2, channels[3]//4, 1, 1, 0)

    def forward(self, x, thermal):

        x1_1a, x1_1b = self.mca1(self.lpr1(x[0]).permute(0,2,3,1)).permute(0,3,1,2), self.mca1t(self.lpt1(thermal[0]).permute(0,2,3,1)).permute(0,3,1,2)
        x2_1a, x2_1b = self.mca2(self.lpr2(x[1]).permute(0,2,3,1)).permute(0,3,1,2), self.mca2t(self.lpt2(thermal[1]).permute(0,2,3,1)).permute(0,3,1,2)
        x3_1a, x3_1b = self.mca3(self.lpr3(x[2]).permute(0,2,3,1)).permute(0,3,1,2), self.mca3t(self.lpt3(thermal[2]).permute(0,2,3,1)).permute(0,3,1,2)
        x4_1a, x4_1b = self.mca4(self.lpr4(x[3]).permute(0,2,3,1)).permute(0,3,1,2), self.mca4t(self.lpt4(thermal[3]).permute(0,2,3,1)).permute(0,3,1,2)

        x1_1lp = torch.nn.Sigmoid()(self.lp1(torch.cat([x1_1a, x1_1b], dim=1)))
        x1_1 = x1_1a*x1_1lp+x1_1b*(1-x1_1lp)


        x2_1lp = torch.nn.Sigmoid()(self.lp2(torch.cat([x2_1a, x2_1b], dim=1)))
        x2_1 = x2_1a * x2_1lp + x2_1b * (1 - x2_1lp)


        x3_1lp = torch.nn.Sigmoid()(self.lp3(torch.cat([x3_1a, x3_1b], dim=1)))
        x3_1 = x3_1a * x3_1lp + x3_1b * (1 - x3_1lp)


        x4_1lp = torch.nn.Sigmoid()(self.lp4(torch.cat([x4_1a, x4_1b], dim=1)))
        x4_1 = x4_1a * x4_1lp + x4_1b * (1 - x4_1lp)

        return x1_1, x2_1, x3_1, x4_1

class Mamba(nn.Module):
    def __init__(self, in_c):
        super(Mamba, self).__init__()
        self.conv = BasicConv(2 * in_c, in_c, 3, 1, 1)
        self.SS2D = SS2D(in_c)
        self.ln = nn.LayerNorm(in_c)

    def forward(self, rgb, t):
        rgbt = self.conv(torch.cat((rgb, t), dim=1))
        out = self.SS2D(self.ln(rgbt.permute(0, 2, 3, 1)))
        out = out.permute(0, 3, 1, 2)
        return out

class Mamba1(nn.Module):
    def __init__(self, in_c):
        super(Mamba1, self).__init__()
        self.SS2D1 = SS2D(in_c)
        self.ln1 = nn.LayerNorm(in_c)
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        rgb = rgb.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        t = t.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        rgbt = torch.stack((rgb, t), dim=2).view(B, -1, C).reshape(B, H, 2*W, C)
        # out = self.SS2D1(self.ln1(rgbt))
        out = self.SS2D1(rgbt)
        out_ = out.view(B, 2*H*W, C)[:, 1::2, :].view(B, H, W, C)
        out = out_.permute(0, 3, 1, 2)
        return out

class Mamba2(nn.Module):
    def __init__(self, in_c, depth):
        super(Mamba2, self).__init__()
        self.depth = depth
        self.SS2D = SS2D(in_c)
        self.ln1 = nn.LayerNorm(in_c)
        self.conv1 = BasicConv(2*in_c, in_c, 3, 1, 1)
        self.conv2 = BasicConv(3*in_c, in_c, 1, 1, 0)
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))
        rgb_ = rgb.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        t_ = t.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        rgbt = torch.stack((rgb_, t_), dim=2).view(B, -1, C).reshape(B, H, 2*W, C)
        out = self.SS2D(rgbt)
        rgb = out.view(B, 2 * H * W, C)[:, 0::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + rgb
        t = out.view(B, 2*H*W, C)[:, 1::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + t
        right = torch.cat((rgb, t), dim=1)
        out = self.conv2(torch.cat((left, right), dim=1))
        return out

class Mamba3(nn.Module):
    def __init__(self, in_c, depth):
        super(Mamba3, self).__init__()
        self.depth = depth
        self.SS2D = SS2D(in_c // 2)
        self.ln1 = nn.LayerNorm(in_c)
        self.conv1 = BasicConv(in_c, in_c // 2, 3, 1, 1)
        self.conv2 = BasicConv(in_c, in_c // 2, 3, 1, 1)
    def forward(self, rgb, t):
        rgb_l, rgb_r = rgb.chunk(2, dim=1)
        t_l, t_r = t.chunk(2, dim=1)
        # left
        left = self.conv1(torch.cat((rgb_l, t_l), dim=1))

        # right
        B, C, H, W = rgb_r.shape
        rgb_r_ = rgb_r.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        t_r_ = t_r.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        rgbt = torch.stack((rgb_r_, t_r_), dim=2).view(B, -1, C).reshape(B, H, 2*W, C)
        out = self.SS2D(rgbt)
        rgb = out.view(B, 2 * H * W, C)[:, 0::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + rgb_r
        t = out.view(B, 2*H*W, C)[:, 1::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + t_r
        right = self.conv2(torch.cat((rgb, t), dim=1))

        output = torch.cat((right, left), dim=1)
        output = channel_shuffle(output, groups=2)
        return output

class Mamba4(nn.Module):
    def __init__(self, in_c, depth):
        super(Mamba4, self).__init__()
        self.depth = depth
        self.SS2D = SS2D_rgbt(in_c)
        self.ln1 = nn.LayerNorm(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c))
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c))
        self.act = nn.SiLU()
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))
        rgb_ = rgb.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        t_ = t.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        rgbt = torch.stack((rgb_, t_), dim=2).view(B, -1, C).reshape(B, H, 2*W, C)
        out = self.SS2D(rgbt)
        rgb = out.view(B, 2 * H * W, C)[:, 0::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + rgb
        t = out.view(B, 2*H*W, C)[:, 1::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + t
        right = torch.cat((rgb, t), dim=1)
        out = self.conv2(self.act(torch.cat((left, right), dim=1)))
        return out

class Mamba5(nn.Module):
    def __init__(self, in_c, depth):
        super(Mamba5, self).__init__()
        self.depth = depth
        self.SS2D = SS2D_rgbt(in_c)
        self.ln1 = nn.LayerNorm(in_c)
        self.conv1 = BasicConv(in_c, in_c, 3, 1, 1)
        self.conv2 = BasicConv(3*in_c, in_c, 1, 1, 0)
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(rgb+t)
        rgb_ = rgb.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        t_ = t.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        rgbt = torch.stack((rgb_, t_), dim=2).view(B, -1, C).reshape(B, H, 2*W, C)
        out = self.SS2D(rgbt)
        rgb = out.view(B, 2 * H * W, C)[:, 0::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + rgb
        t = out.view(B, 2*H*W, C)[:, 1::2, :].view(B, H, W, C).permute(0, 3, 1, 2) + t
        right = torch.cat((rgb, t), dim=1)
        out = self.conv2(torch.cat((left, right), dim=1))
        return out
#################################################################################
#                             Basic Layers                                     #
#################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class MultiscalseLayer(nn.Module):
    def __init__(self, in_c, out_c, scales):
        super(MultiscalseLayer, self).__init__()
        self.scale = nn.ModuleList()
        for i in scales:
            self.scale.append(BasicConv(in_c, in_c // 4, i, 1, get_same_padding(i)))
        self.conv = BasicConv(in_c, in_c // 2, 1, 1, 0)
    def forward(self, rgb, t):
        scale_feature = []
        rgbt = torch.cat((rgb, t), dim=1)
        for i in range(4):
            scale_feature.append(self.scale[i](rgbt))
        out = self.conv(torch.cat(scale_feature, dim=1))
        return out

class DSMultiscalseLayer(nn.Module):
    def __init__(self, in_c, out_c, scales):
        super(DSMultiscalseLayer, self).__init__()
        self.scale = nn.ModuleList()
        for i in scales:
            self.scale.append(DSConv(in_c, in_c // 8, i, 1, get_same_padding(i)))
        self.conv = BasicConv(in_c // 2, in_c // 2, 1, 1, 0)
    def forward(self, rgb, t):
        scale_feature = []
        rgbt = torch.cat((rgb, t), dim=1)
        for i in range(4):
            scale_feature.append(self.scale[i](rgbt))
        out = self.conv(torch.cat(scale_feature, dim=1))
        return out

#################################################################################
#                             Basic Blocks                                      #
#################################################################################

class SS2D_rgbt(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm1 = nn.LayerNorm(self.d_inner)
        self.out_norm2 = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.FFN = Mlp(self.d_model, self.d_model * 4)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4
        # 核心，进行横向检索与竖向检索
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # 进行正向检索与反向检索
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        x = self.in_proj(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm1(y)
        out = self.out_proj(y)
        # FFN
        out = self.FFN(self.out_norm2(out)) + out
        return out

class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6.0,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        ## project rgbt, rgb, t to kv, q_rgb, q_t ##
        self.kv = ConvLayer(
            in_channels,
            4 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.q_rgb = ConvLayer(
            in_channels,
            total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.q_t = ConvLayer(
            in_channels,
            total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg_rgbt = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        4 * total_dim,
                        4 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=4 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(4 * total_dim, 4 * total_dim, 1, groups=4 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.aggreg_rgb = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        total_dim,
                        total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(total_dim, total_dim, 1, groups=heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.aggreg_t = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        total_dim,
                        total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(total_dim, total_dim, 1, groups=heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            2 * total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @autocast(enabled=False)
    def relu_linear_att(self, kv, q):
        B, _, H, W = list(kv.size())

        if kv.dtype == torch.float16:
            kv = kv.float()
        if q.dtype == torch.float16:
            q = q.float()
        kv = torch.reshape(kv, (B, -1, 2 * self.dim, H * W, ),)
        q = torch.reshape(q, (B, -1, self.dim, H * W, ), )
        kv = torch.transpose(kv, -1, -2)
        q = torch.transpose(q, -1, -2)
        k, v = (
            kv[..., 0:self.dim],
            kv[..., self.dim:],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, rgbt, rgb, t):
        # generate multi-scale q, k, v
        B, C, H, W = rgb.size()
        kv = self.kv(rgbt)
        q_t = self.q_t(t)
        q_rgb = self.q_rgb(rgb)
        multi_scale_kv = [kv]
        multi_scale_q_t = [q_t]
        multi_scale_q_rgb = [q_rgb]
        for op1, op2, op3 in zip(self.aggreg_rgbt, self.aggreg_t, self.aggreg_rgb):
            multi_scale_kv.append(op1(kv))
            multi_scale_q_t.append(op2(q_t))
            multi_scale_q_rgb.append(op3(q_rgb))
        multi_scale_kv1 = []
        multi_scale_kv2 = []
        for i in multi_scale_kv:
            multi_scale_kv1.append(i[:, :2*C, :, :])
            multi_scale_kv2.append(i[:, 2*C:, :, :])
        multi_scale_kv1 = torch.cat(multi_scale_kv1, dim=1)
        multi_scale_kv2 = torch.cat(multi_scale_kv2, dim=1)
        multi_scale_q_t = torch.cat(multi_scale_q_t, dim=1)
        multi_scale_q_rgb = torch.cat(multi_scale_q_rgb, dim=1)
        out1 = self.relu_linear_att(multi_scale_kv1, multi_scale_q_t)
        out2 = self.relu_linear_att(multi_scale_kv2, multi_scale_q_rgb)
        out = self.proj(torch.cat((out1, out2), dim=1))

        return out

class ResBlock(nn.Module):
    def __init__(self, in_c, ratio=4):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c // ratio, 1)
        self.norm1 = nn.BatchNorm2d(in_c // ratio)
        self.conv2 = nn.Conv2d(in_c // ratio, in_c // ratio, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(in_c // ratio)
        self.conv3 = nn.Conv2d(in_c // ratio, in_c // 2, 1)
        self.norm3 = nn.BatchNorm2d(in_c // 2)
        self.relu = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(in_c, in_c // 2, 1)
        self.norm4 = nn.BatchNorm2d(in_c // 2)
    def forward(self, x):
        x_ = self.relu(self.norm1(self.conv1(x)))
        x_ = self.relu(self.norm2(self.conv2(x_)))
        out = self.relu(self.norm4(self.conv4(x))+self.norm3(self.conv3(x_)))
        return out

class BasicConv(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(0.2)
    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        # out = self.drop(out)
        return out

class DSC(nn.Module):
    def __init__(self, inchannels, outchannels, kenelsize, padding, dilation):
        super(DSC, self).__init__()
        self.depthwiseConv = nn.Conv2d(inchannels, inchannels, kenelsize, groups=inchannels, padding=padding, dilation=dilation)
        self.pointwiseConv = nn.Conv2d(inchannels, outchannels, 1)
        self.BN = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.depthwiseConv(x)
        x = self.pointwiseConv(x)
        x = self.relu(self.BN(x))
        return x

class Dilation(nn.Module):
    def __init__(self, in_c):
        super(Dilation, self).__init__()
        self.dilation1 = DSC(in_c, in_c // 8, 3, 3, 3)
        self.dilation2 = DSC(in_c, in_c // 8, 3, 5, 5)
        self.dilation3 = DSC(in_c, in_c // 8, 3, 7, 7)
        self.dilation4 = DSC(in_c, in_c // 8, 3, 9, 9)
        self.conv = nn.Conv2d(in_c // 2, in_c // 2, 1, 1, 0)
    def forward(self, rgb, t):
        input = torch.cat((rgb, t), dim=1)
        out1 = self.dilation1(input)
        out2 = self.dilation2(input)
        out3 = self.dilation3(input)
        out4 = self.dilation4(input)
        out = self.conv(torch.cat((out1, out2, out3, out4), dim=1))
        return out

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x