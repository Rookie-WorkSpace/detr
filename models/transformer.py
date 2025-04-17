# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer类。

从torch.nn.Transformer复制并修改:
    * 位置编码在多头注意力中传递
    * 移除了编码器末尾的额外LayerNorm
    * 解码器返回所有解码层的激活堆栈
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    DETR的Transformer模型。
    包含一个编码器和一个解码器,用于处理图像特征序列和目标查询。
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """
        初始化Transformer。
        
        参数:
            d_model: 模型的维度,默认512
            nhead: 多头注意力的头数,默认8
            num_encoder_layers: 编码器层数,默认6
            num_decoder_layers: 解码器层数,默认6
            dim_feedforward: 前馈网络的维度,默认2048
            dropout: dropout率,默认0.1
            activation: 激活函数,默认"relu"
            normalize_before: 是否在注意力和FFN之前进行归一化,默认False
            return_intermediate_dec: 是否返回解码器中间层的输出,默认False
        """
        super().__init__()

        # 构建编码器层
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 构建解码器层
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 初始化参数
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        使用Xavier均匀分布初始化所有参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        前向传播函数。
        
        参数:
            src: 输入特征图 [batch_size, channels, height, width]
            mask: 输入特征的padding mask [batch_size, height*width]
            query_embed: 目标查询嵌入 [num_queries, embed_dim]
            pos_embed: 位置编码 [batch_size, embed_dim, height, width]
            
        返回:
            hs: 解码器输出 [num_layers, batch_size, num_queries, embed_dim]
            memory: 编码器输出 [batch_size, channels, height, width]
        """
        # 将NxCxHxW展平为HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        # 初始化目标为全零
        tgt = torch.zeros_like(query_embed)
        # 通过编码器
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 通过解码器
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # 调整输出维度顺序并返回
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """
    Transformer编码器。
    由多个编码器层堆叠而成。
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        初始化编码器。
        
        参数:
            encoder_layer: 单个编码器层
            num_layers: 编码器层数
            norm: 归一化层,默认None
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        前向传播函数。
        
        参数:
            src: 输入序列
            mask: 注意力mask
            src_key_padding_mask: key padding mask
            pos: 位置编码
            
        返回:
            output: 编码器输出
        """
        output = src

        # 依次通过每个编码器层
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Transformer解码器。
    由多个解码器层堆叠而成。
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        初始化解码器。
        
        参数:
            decoder_layer: 单个解码器层
            num_layers: 解码器层数
            norm: 归一化层,默认None
            return_intermediate: 是否返回中间层输出,默认False
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        前向传播函数。
        
        参数:
            tgt: 目标序列
            memory: 编码器输出的memory
            tgt_mask: 目标序列的注意力mask
            memory_mask: memory的注意力mask
            tgt_key_padding_mask: 目标序列的key padding mask
            memory_key_padding_mask: memory的key padding mask
            pos: 位置编码
            query_pos: 查询位置编码
            
        返回:
            output: 解码器输出
        """
        output = tgt

        intermediate = []

        # 依次通过每个解码器层
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # 如果需要归一化,则进行归一化
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层。
    包含自注意力层和前馈网络。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        初始化编码器层。
        
        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度,默认2048
            dropout: dropout率,默认0.1
            activation: 激活函数,默认"relu"
            normalize_before: 是否在前进行归一化,默认False
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before    # 是否在前进行归一化

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码加到输入张量上
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        后归一化的前向传播
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        前归一化的前向传播
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        根据normalize_before选择前向传播方式
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层。
    包含自注意力层、交叉注意力层和前馈网络。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        初始化解码器层。
        
        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度,默认2048
            dropout: dropout率,默认0.1
            activation: 激活函数,默认"relu"
            normalize_before: 是否在前进行归一化,默认False
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 自注意力层
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 交叉注意力层
        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward) # 线性层1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model) # 线性层2

        self.norm1 = nn.LayerNorm(d_model) # 归一化层1
        self.norm2 = nn.LayerNorm(d_model) # 归一化层2
        self.norm3 = nn.LayerNorm(d_model) # 归一化层3
        self.dropout1 = nn.Dropout(dropout) # 丢弃层1
        self.dropout2 = nn.Dropout(dropout) # 丢弃层2
        self.dropout3 = nn.Dropout(dropout) # 丢弃层3

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码加到输入张量上
        """
        return tensor if pos is None else tensor + pos  # 如果pos为None,则返回tensor,否则返回tensor+pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        后归一化的前向传播

        参数:
            tgt: 目标序列
            memory: 编码器输出的memory
            tgt_mask: 目标序列的注意力mask
            memory_mask: memory的注意力mask
            tgt_key_padding_mask: 目标序列的key padding mask
            memory_key_padding_mask: memory的key padding mask
            pos: 位置编码
            query_pos: 查询位置编码
        """
        q = k = self.with_pos_embed(tgt, query_pos)  # 位置编码
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # 自注意力层
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  # 交叉注意力层
        tgt = tgt + self.dropout2(tgt2)  # 丢弃层2
        tgt = self.norm2(tgt)  # 归一化层2
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # 线性层2
        tgt = tgt + self.dropout3(tgt2)  # 丢弃层3
        tgt = self.norm3(tgt)  # 归一化层3
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        前归一化的前向传播

        参数:
            tgt: 目标序列
            memory: 编码器输出的memory
            tgt_mask: 目标序列的注意力mask
            memory_mask: memory的注意力mask
            tgt_key_padding_mask: 目标序列的key padding mask
            memory_key_padding_mask: memory的key padding mask
            pos: 位置编码
            query_pos: 查询位置编码
        """
        tgt2 = self.norm1(tgt)  # 归一化层1
        q = k = self.with_pos_embed(tgt2, query_pos)  # 位置编码
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # 自注意力层
        tgt = tgt + self.dropout1(tgt2)  # 丢弃层1
        tgt2 = self.norm2(tgt)  # 归一化层2
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        根据normalize_before选择前向传播方式

        参数:
            tgt: 目标序列
            memory: 编码器输出的memory
            tgt_mask: 目标序列的注意力mask
            memory_mask: memory的注意力mask
            tgt_key_padding_mask: 目标序列的key padding mask
            memory_key_padding_mask: memory的key padding mask
            pos: 位置编码
            query_pos: 查询位置编码
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    """
    克隆模块N次
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """
    根据参数构建Transformer

    参数:
        args: 配置参数
    """
    return Transformer(
        d_model=args.hidden_dim,    # 模型维度
        dropout=args.dropout,      # 丢弃率
        nhead=args.nheads,         # 注意力头数
        dim_feedforward=args.dim_feedforward, # 前馈网络维度
        num_encoder_layers=args.enc_layers, # 编码器层数
        num_decoder_layers=args.dec_layers, # 解码器层数
        normalize_before=args.pre_norm, # 是否在前进行归一化
        return_intermediate_dec=True, # 是否返回中间层输出
    )


def _get_activation_fn(activation):
    """
    根据字符串返回对应的激活函数
    
    参数:
        activation: 激活函数名称
        
    返回:
        激活函数
    """
    if activation == "relu":    # 如果激活函数为relu,则返回relu
        return F.relu
    if activation == "gelu":    # 如果激活函数为gelu,则返回gelu
        return F.gelu
    if activation == "glu":    # 如果激活函数为glu,则返回glu
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")    # 如果激活函数不是relu/gelu/glu,则抛出异常
