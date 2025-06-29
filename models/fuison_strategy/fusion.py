import torch
import torch.nn as nn
from jinja2.utils import concat
from torch import Tensor

import torch.nn.functional as F

from timm.layers import trunc_normal_
import math

from einops import rearrange, repeat
from model_others.RGB_T.CMX.models.net_utils import FeatureFusionModule as FFM
from model_others.RGB_T.CMX.models.net_utils import FeatureRectifyModule as FRM
from proposed.attention_module import FeatureFusionModule
from model_others.RGB_T.MAINet import TSFA
from model_others.RGB_T.MDNet.model import MultiSpectralAttentionLayer, SS_Conv_SSM
from backbone.MedMamba import SS2D
from model_others.RGB_T.sigma.encoders.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class Fusion_Module(nn.Module):
    def __init__(self, fusion_mode, channels, num_heads=[1, 2, 4, 8], norm_fuse=nn.BatchNorm2d):
        super(Fusion_Module, self).__init__()
        self.fusion_mode = fusion_mode
        self.norm_layer = nn.ModuleList()
        self.fusion = nn.ModuleList()

        if self.fusion_mode == 'cat':
            for i in channels:
                self.fusion.append(BasicConv(2 * i, i, 3, 1, 1))

        if self.fusion_mode == 'demo1':
            for i in channels:
                self.fusion.append(Demo1(i))

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

        if self.fusion_mode == 'CDA':
            self.CDAs = nn.ModuleList([
                FeatureFusionModule(channels[0]),
                FeatureFusionModule(channels[1]),
                FeatureFusionModule(channels[2]),
                FeatureFusionModule(channels[3]),
            ])
            # self.CDA = FeatureFusionModule(channels[3])

        if self.fusion_mode == 'sigma':
            self.CroMB = nn.ModuleList()
            self.ConMB = nn.ModuleList()
            for i in channels:
                self.CroMB.append(CrossMambaFusionBlock(hidden_dim=i,
                                      mlp_ratio=0.0,
                                      d_state=4, ))
                self.ConMB.append(ConcatMambaFusionBlock(hidden_dim=i,
                                                                mlp_ratio=0.0,
                                                                d_state=4,))

            self.FFMs = nn.ModuleList([
                FFM(dim=channels[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
                FFM(dim=channels[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
                FFM(dim=channels[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
                FFM(dim=channels[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        # if self.fusion_mode == 'MDFusion':
        #     self.fusion = MDFusion(channels)

        if self.fusion_mode == 'TSFA':
            for channel in channels:
                self.fusion.append(TSFA(channel))

        if self.fusion_mode == 'CM-SSM':
            for channel in channels:
                self.fusion.append(CM_SSM(channel))

        if self.fusion_mode == 'M-SSM':
            for channel in channels:
                self.fusion.append(M_SSM(channel))





    def forward(self, rgb, t):
        outs = []
        if self.fusion_mode == 'MDFusion':
            outs = self.fusion(rgb, t)

        for i in range(4):
            if self.fusion_mode == 'add':
                outs.append(rgb[i] + t[i])

            if self.fusion_mode == 'max':
                out, _ = torch.max(torch.stack([rgb[i], t[i]], dim=1), dim=1)
                outs.append(out)

            if self.fusion_mode == 'demo1':
                outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode == 'CMX':
                rgb_, t_ = self.FRMs[i](rgb[i], t[i])
                # rgb_, t_ = rgb[i], t[i]
                out = self.FFMs[i](rgb_, t_)
                outs.append(out)

            if self.fusion_mode == 'TSFA':
                outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode == 'CDA':
                if i >= 1:
                    outs.append(self.CDAs[i](rgb[i], t[i]))
                else:
                    outs.append(rgb[i]+t[i])
                # else:
                #     outs.append(self.CDAs[i](rgb[i], t[i]))
                # outs.append(self.CDAs[i](rgb[i], t[i]))

            if self.fusion_mode == 'cat':
                out= self.fusion[i](torch.cat((rgb[i], t[i]), dim=1))
                outs.append(out)


            # Mamba4 is best
            if self.fusion_mode in ['CM-SSM', 'M-SSM']:
                outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode == 'sigma':
                rgb_, t_ = self.CroMB[i](rgb[i].permute(0, 2, 3, 1), t[i].permute(0, 2, 3, 1))
                out = self.ConMB[i](rgb_, t_).permute(0, 3, 1, 2)
                outs.append(out)

        return outs

class CM_SSM(nn.Module):
    def __init__(self, in_c):
        super(CM_SSM, self).__init__()
        self.SS2D = SS2D_rgbt(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_, t_ = self.SS2D(rgb.permute(0, 2, 3, 1), t.permute(0, 2, 3, 1))
        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
        return out

class M_SSM(nn.Module):
    def __init__(self, in_c):
        super(M_SSM, self).__init__()
        self.SS2D_rgb = SS2D(in_c)
        self.SS2D_t = SS2D(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_ = self.SS2D_rgb(rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        t_ = self.SS2D_t(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb, t), dim=1))
        return out

class Demo1(nn.Module):
    def __init__(self, in_c):
        super(Demo1, self).__init__()
        self.conv1 = BasicConv(in_c * 2, in_c, 3, 1, 1)
        self.conv2 = BasicConv(in_c* 2, in_c, 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        path1 = self.conv1(torch.cat((rgb, t), dim=1))
        avg = self.avgpool(torch.cat((rgb, t), dim=1))
        max = self.maxpool(torch.cat((rgb, t), dim=1))
        path2 = torch.mul(torch.cat((rgb, t), dim=1), self.sigmoid(avg+max))
        path2 = self.conv2(path2)
        fusion = path1 + path2
        return fusion



#################################################################################
#                             Basic Layers                                      #
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

        self.in_proj1 = nn.Linear(self.d_model, 2*self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, 2*self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d2 = nn.Conv2d(
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

        self.norm1 = nn.LayerNorm(self.d_inner)
        self.norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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

    def forward_corev0(self, rgb: torch.Tensor, t: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = rgb.shape
        L = H * W
        K = 4
        # 核心，进行横向检索与竖向检索
        rgb_hwwh = torch.stack([rgb.view(B, -1, L), torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        t_hwwh = torch.stack([t.view(B, -1, L), torch.transpose(t, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                               dim=1).view(B, 2, -1, L)
        x_hwwh = torch.stack((rgb_hwwh, t_hwwh), dim=-1).view(B, 2, -1, 2*L)
        # 进行正向检索与反向检索
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, 2*L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, 2*L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, 2*L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, 2*L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, 2*L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, 2*L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, 2*L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, 2*L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, 2*L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, 2*L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, rgb: torch.Tensor, t: torch.Tensor, **kwargs):
        B, H, W, C = rgb.shape
        rgb = self.in_proj1(rgb)
        rgb1, rgb2 = rgb.chunk(2, dim=-1)
        rgb1 = rgb1.permute(0, 3, 1, 2).contiguous()
        rgb1 = self.act(self.conv2d1(rgb1))  # (b, d, h, w)

        t = self.in_proj2(t)
        t1, t2 = t.chunk(2, dim=-1)
        t1 = t1.permute(0, 3, 1, 2).contiguous()
        t1 = self.act(self.conv2d1(t1))

        y1, y2, y3, y4 = self.forward_core(rgb1, t1)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        rgb1 = y[:, 0::2, :].permute(0, 1, 2).contiguous().view(B, H, W, -1)
        rgb_ = self.norm1(rgb1) * F.silu(rgb2)
        rgb_out = self.out_proj1(rgb_).permute(0, 3, 1, 2)

        t1 = y[:, 1::2, :].permute(0, 1, 2).contiguous().view(B, H, W, -1)
        t_ = self.norm2(t1) * F.silu(t2)
        t_out = self.out_proj2(t_).permute(0, 3, 1, 2)

        return rgb_out, t_out

#################################################################################
#                             Basic Functions                                   #
#################################################################################

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

#################################################################################
#                             Comparison Methods                                #
#################################################################################

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