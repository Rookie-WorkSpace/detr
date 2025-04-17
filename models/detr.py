# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR模型和损失函数类。
"""
# 导入PyTorch相关包
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口,提供各种神经网络层的函数实现
from torch import nn  # PyTorch神经网络模块

# 导入工具函数
from util import box_ops  # 边界框操作相关函数
from util.misc import (NestedTensor,  # 嵌套张量数据结构,用于处理不同尺寸的图像batch
                       nested_tensor_from_tensor_list,  # 将张量列表转换为嵌套张量
                       accuracy,  # 计算分类准确率
                       get_world_size,  # 获取分布式训练的world size(进程数)
                       interpolate,  # 插值操作,用于调整特征图大小
                       is_dist_avail_and_initialized)  # 检查分布式环境是否可用和初始化

# 导入模型组件
from .backbone import build_backbone  # 构建主干网络(如ResNet)用于提取图像特征
from .matcher import build_matcher  # 构建匹配器(匹配预测框和真实框)
from .segmentation import (DETRsegm,  # DETR分割模型,在检测基础上添加分割头
                           PostProcessPanoptic,  # 全景分割后处理,生成最终分割结果
                           PostProcessSegm,  # 实例分割后处理
                           dice_loss,  # Dice损失函数,用于分割任务
                           sigmoid_focal_loss)  # Sigmoid focal损失函数,用于处理类别不平衡
from .transformer import build_transformer  # 构建Transformer编码器-解码器架构


class DETR(nn.Module):
    """ 这是执行目标检测的DETR模块 """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ 初始化模型。
        参数:
            backbone: 主干网络模块。参见backbone.py
            transformer: transformer架构模块。参见transformer.py
            num_classes: 目标类别数
            num_queries: 目标查询数,即检测槽位。这是DETR在单张图像中可以检测的最大目标数。
                        对于COCO数据集,我们建议使用100个查询。
            aux_loss: 是否使用辅助解码损失(在每个解码器层计算损失)。
        """
        super().__init__()
        self.num_queries = num_queries  # 存储查询数
        self.transformer = transformer  # Transformer模块
        hidden_dim = transformer.d_model  # 获取Transformer的隐藏维度
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # 分类头,+1是为了表示"无目标"类
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 边界框回归头,使用3层MLP
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # 可学习的查询嵌入
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # 输入投影层,调整通道数
        self.backbone = backbone  # 主干网络
        self.aux_loss = aux_loss  # 是否使用辅助损失

    def forward(self, samples: NestedTensor):
        """ 前向传播函数。
        输入参数samples是一个NestedTensor,包含:
               - samples.tensor: 批量图像, 形状为 [batch_size x 3 x H x W]
               - samples.mask: 二值掩码, 形状为 [batch_size x H x W], 在填充像素处为1

        返回一个字典,包含以下元素:
               - "pred_logits": 所有查询的分类logits(包括无目标类)。
                                形状= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": 所有查询的归一化框坐标。
                               表示为(center_x, center_y, height, width)。这些值在[0, 1]范围内归一化,
                               相对于每个图像的尺寸(不考虑可能的填充)。
                               参见PostProcess了解如何获取未归一化的边界框。
               - "aux_outputs": 可选,仅在激活辅助损失时返回。这是一个字典列表,
                                每个字典包含上述两个键,对应每个解码器层。
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  # 将输入转换为NestedTensor格式
        features, pos = self.backbone(samples)  # 通过主干网络提取特征和位置编码

        src, mask = features[-1].decompose()  # 分解最后一层特征
        assert mask is not None
        # 通过Transformer处理特征
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # 预测类别和边界框
        outputs_class = self.class_embed(hs)  # 分类预测
        outputs_coord = self.bbox_embed(hs).sigmoid()  # 边界框预测并归一化到[0,1]
        # 构建输出字典
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个变通方法,使torchscript满意,因为torchscript
        # 不支持包含非同质值的字典,比如同时包含张量和列表的字典。
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ 这个类计算DETR的损失。
    处理过程分两步:
        1) 使用匈牙利算法计算真实框和模型输出之间的匹配
        2) 对每对匹配的真实值/预测值进行监督(监督类别和边界框)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ 创建损失函数。
        参数:
            num_classes: 目标类别数,不包括特殊的"无目标"类别
            matcher: 能够计算目标和提议之间匹配的模块
            weight_dict: 字典,键为损失名称,值为相应的权重
            eos_coef: 应用于"无目标"类别的相对分类权重
            losses: 要应用的所有损失列表。参见get_loss获取可用损失列表。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # 创建类别权重,最后一个类别(无目标)有特殊的权重
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """分类损失(NLL)
        targets字典必须包含键"labels",其值为维度[nb_target_boxes]的张量
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 计算交叉熵损失
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO 这可能应该是一个单独的损失,而不是在这里硬编码
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ 计算基数误差,即预测的非空框数量的绝对误差
        这不是真正的损失,它仅用于记录目的。它不传播梯度
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # 计算预测中不是"无目标"的数量(最后一个类别)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算与边界框相关的损失,包括L1回归损失和GIoU损失
           targets字典必须包含键"boxes",其值为维度[nb_target_boxes, 4]的张量
           目标框的格式应为(center_x, center_y, w, h),并按图像大小归一化。
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 计算GIoU损失
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """计算与掩码相关的损失:focal loss和dice loss。
           targets字典必须包含键"masks",其值为维度[nb_target_boxes, h, w]的张量
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO 使用valid来掩盖由于填充导致的无效区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # 将预测上采样到目标尺寸
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # 根据匹配索引重排预测
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 根据匹配索引重排目标
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 损失函数映射表
        loss_map = {
            'labels': self.loss_labels,  # 分类损失
            'cardinality': self.loss_cardinality,  # 基数损失
            'boxes': self.loss_boxes,  # 边界框损失
            'masks': self.loss_masks  # 掩码损失
        }
        assert loss in loss_map, f'你确定要计算{loss}损失吗?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ 执行损失计算。
        参数:
             outputs: 张量字典,参见模型输出规范了解格式
             targets: 字典列表,长度等于batch_size。
                      每个字典中的预期键取决于应用的损失,参见每个损失的文档
        """
        # 移除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 检索最后一层输出和目标之间的匹配
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点的平均目标框数量,用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果有辅助损失,对每个中间层的输出重复此过程
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # 中间掩码损失计算成本太高,我们忽略它们
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 仅为最后一层启用日志记录
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ 此模块将模型的输出转换为COCO API期望的格式"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ 执行计算
        参数:
            outputs: 模型的原始输出
            target_sizes: 维度为[batch_size x 2]的张量,包含每个图像的尺寸
                          对于评估,这必须是原始图像尺寸(在任何数据增强之前)
                          对于可视化,这应该是数据增强后但在填充之前的图像尺寸
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 计算每个类别的概率和最大概率的类别
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # 将框格式转换为[x0, y0, x1, y1]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # 从相对[0, 1]坐标转换为绝对[0, height]坐标
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # 构建结果列表
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ 非常简单的多层感知机(也称为FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 构建线性层列表
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 除最后一层外都使用ReLU激活
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # `num_classes`的命名这里有点误导。
    # 它实际上对应于`max_obj_id + 1`,其中max_obj_id
    # 是数据集中类别的最大ID。例如,
    # COCO的max_obj_id是90,所以我们传入`num_classes`为91。
    # 另一个例子,对于只有ID为1的单个类别的数据集,
    # 你应该传入`num_classes`为2 (max_obj_id + 1)。
    # 更多详细信息,请查看以下讨论
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # 对于全景分割,我们只需添加一个足够大的num_classes来容纳
        # max_obj_id + 1,但具体值并不重要
        num_classes = 250
    device = torch.device(args.device)

    # 构建模型组件
    backbone = build_backbone(args)  # 构建主干网络
    transformer = build_transformer(args)  # 构建transformer

    # 构建DETR模型
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    # 如果需要分割,添加分割头
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    # 构建匹配器和损失权重
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO 这是一个临时解决方案
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 定义损失列表
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    
    # 构建损失函数和后处理模块
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
