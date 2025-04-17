# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
主干网络模块。
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d的特殊版本,其中批量统计信息和仿射参数都被冻结。

    从torchvision.misc.ops复制粘贴,并在rqsrt之前添加了eps参数。
    如果没有这个eps,除了torchvision.models.resnet[18,34,50,101]之外的任何其他模型都会产生nan值。
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 注册固定的缓冲区,这些参数在训练时不会更新
        self.register_buffer("weight", torch.ones(n))  # 缩放参数gamma
        self.register_buffer("bias", torch.zeros(n))   # 偏置参数beta
        self.register_buffer("running_mean", torch.zeros(n))  # 运行时均值
        self.register_buffer("running_var", torch.ones(n))    # 运行时方差

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 从状态字典加载参数时,删除num_batches_tracked键(因为我们不需要它)
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # 将所有reshape操作移到开始,使其更易于融合优化
        w = self.weight.reshape(1, -1, 1, 1)      # [1,C,1,1]形状
        b = self.bias.reshape(1, -1, 1, 1)        # [1,C,1,1]形状
        rv = self.running_var.reshape(1, -1, 1, 1)  # [1,C,1,1]形状
        rm = self.running_mean.reshape(1, -1, 1, 1)  # [1,C,1,1]形状
        eps = 1e-5  # 防止除零的小常数
        scale = w * (rv + eps).rsqrt()  # 计算缩放因子
        bias = b - rm * scale           # 计算偏置
        return x * scale + bias         # 应用批归一化


class BackboneBase(nn.Module):
    """主干网络的基类,处理特征提取"""

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # 选择性地冻结主干网络的参数
        for name, parameter in backbone.named_parameters():
            # 如果不训练主干网络,或者参数不在layer2/3/4中,则冻结参数
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        # 设置需要返回的层
        if return_interm_layers:
            # 如果需要中间层,返回所有层
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # 否则只返回最后一层
            return_layers = {'layer4': "0"}
        # 使用IntermediateLayerGetter来获取指定层的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # 通过主干网络处理输入
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        # 为每个输出特征图生成对应的mask
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将mask插值到与特征图相同的大小
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet主干网络,使用冻结的BatchNorm"""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 构建ResNet模型
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],  # 设置空洞卷积
            pretrained=is_main_process(),  # 是否使用预训练权重
            norm_layer=FrozenBatchNorm2d  # 使用冻结的BatchNorm
        )
        # 根据ResNet类型设置通道数
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """将主干网络和位置编码组合在一起的模块"""
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # 首先通过主干网络
        xs = self[0](tensor_list)   # 通过主干网络处理输入
        out: List[NestedTensor] = []    # 存储主干网络的输出
        pos = []
        # 为每个特征层计算位置编码
        for name, x in xs.items():
            out.append(x)
            # 计算位置编码并确保其数据类型与特征张量匹配
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    构建完整的主干网络
    
    参数:
        args: 配置参数,包含:
            - lr_backbone: 主干网络的学习率
            - masks: 是否返回中间层特征
            - backbone: 主干网络的类型(如'resnet50')
            - dilation: 是否使用空洞卷积
    """
    # 构建位置编码模块
    position_embedding = build_position_encoding(args)
    # 根据学习率决定是否训练主干网络
    train_backbone = args.lr_backbone > 0
    # 是否返回中间层特征(用于mask预测)
    return_interm_layers = args.masks
    # 构建主干网络
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 将主干网络和位置编码组合
    model = Joiner(backbone, position_embedding)
    # 保存输出通道数
    model.num_channels = backbone.num_channels
    return model
