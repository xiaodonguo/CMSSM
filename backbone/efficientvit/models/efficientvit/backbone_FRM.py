# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from backbone.efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
from backbone.efficientvit.models.utils import build_kwargs_from_config
from model_others.RGB_T.CMX.models.net_utils import FeatureRectifyModule as FRM


__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
]


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem1 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]

        self.input_stem2 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]

        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem1.append(ResidualBlock(block, IdentityLayer()))

            # block2 = self.build_local_block(
            #     in_channels=width_list[0],
            #     out_channels=width_list[0],
            #     stride=1,
            #     expand_ratio=1,
            #     norm=norm,
            #     act_func=act_func,
            # )
            self.input_stem2.append(ResidualBlock(block, IdentityLayer()))


        in_channels = width_list[0]
        self.input_stem1 = OpSequential(self.input_stem1)
        self.input_stem2 = OpSequential(self.input_stem2)
        self.width_list.append(in_channels)

        # stages
        self.stages1 = []
        self.stages2 = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage1 = []
            stage2 = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage1.append(block)

                # block2 = self.build_local_block(
                #     in_channels=in_channels,
                #     out_channels=w,
                #     stride=stride,
                #     expand_ratio=expand_ratio,
                #     norm=norm,
                #     act_func=act_func,
                # )
                # block2 = ResidualBlock(block2, IdentityLayer() if stride == 1 else None)
                stage2.append(block)
                in_channels = w

            self.stages1.append(OpSequential(stage1))
            self.stages2.append(OpSequential(stage2))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage1 = []
            stage2 = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            # block2 = self.build_local_block(
            #     in_channels=in_channels,
            #     out_channels=w,
            #     stride=2,
            #     expand_ratio=expand_ratio,
            #     norm=norm,
            #     act_func=act_func,
            #     fewer_norm=True,
            # )
            stage1.append(ResidualBlock(block, None))
            stage2.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage1.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
                stage2.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages1.append(OpSequential(stage1))
            self.stages2.append(OpSequential(stage2))
            self.width_list.append(in_channels)
        self.stages1 = nn.ModuleList(self.stages1)
        self.stages2 = nn.ModuleList(self.stages2)
        self.FRMs = nn.ModuleList()
        for dim in width_list[1:]:
            self.FRMs.append(FRM(dim))

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)['state_dict']
            # print(checkpoint)
            new_checkpoint = OrderedDict()
            for key, value in checkpoint.items():
                if 'backbone.input_stem' in key:
                    new_key_stem1 = key.replace('backbone.input_stem', 'input_stem1')
                    new_key_stem2 = key.replace('backbone.input_stem', 'input_stem2')
                    new_checkpoint[new_key_stem1] = value
                    new_checkpoint[new_key_stem2] = value
                if 'backbone.stages' in key:
                    new_key_stage1 = key.replace('backbone.stages', 'stages1')
                    new_key_stage2 = key.replace('backbone.stages', 'stages2')
                    new_checkpoint[new_key_stage1] = value
                    new_checkpoint[new_key_stage2] = value
            missing_keys, unexpected_keys = self.load_state_dict(new_checkpoint, strict=False)
            print('pretrained weights have been loaded')
            # 打印未匹配的键
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            print('pretrained weights have been loaded')
        elif pretrained is None:
            print('without pretrained')
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, rgb, t):

        rgbs = []
        ts = []
        rgb = self.input_stem1(rgb)
        t = self.input_stem2(t)
        for stage_id, (stage1, stage2, FRM) in enumerate(zip(self.stages1, self.stages2, self.FRMs)):
            rgb = stage1(rgb)
            t = stage2(t)
            rgb, t = FRM(rgb, t)
            rgbs.append(rgb)
            ts.append(t)
        return rgbs, ts


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                stage_id=0,
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:4], depth_list[1:4]), start=1):
            stage = []
            for i in range(d + 1):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    stage_id=stage_id,
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4 if stride == 1 else 16,
                    norm=norm,
                    act_func=act_func,
                    fewer_norm=stage_id > 2,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[4:], depth_list[4:]), start=4):
            stage = []
            block = self.build_local_block(
                stage_id=stage_id,
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=24,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=qkv_dim,
                        expand_ratio=6,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        stage_id: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif stage_id <= 2:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)['state_dict']
            # print(checkpoint)
            new_checkpoint = OrderedDict()
            for key, value in checkpoint.items():
                # 将 'backbone.' 前缀去掉
                new_key = key.replace('backbone.', '')
                new_checkpoint[new_key] = value
            missing_keys, unexpected_keys = self.load_state_dict(new_checkpoint, strict=False)
            print('pretrained weights have been loaded')
            # 打印未匹配的键
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            print('pretrained weights have been loaded')
        elif pretrained is None:
            print('without pretrained')
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # output_dict = {"input": x}
        # for stage_id, stage in enumerate(self.stages):
        #     output_dict["stage%d" % stage_id] = x = stage(x)
        # output_dict["stage_final"] = x
        # return output_dict
        outputs = []
        for stage_id, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
        return outputs[1:]


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone
