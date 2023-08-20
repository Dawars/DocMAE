"""
This code is a modified version of DocTr where the transformer module is replaced by the original transformer model.
MIT Licence https://github.com/fh2019ustc/DocTr
"""
import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from docmae.models.transformer import BasicEncoder, build_position_encoding


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, tgt):
        q = k = v = tgt
        tgt2 = self.self_attn(q, k, v, need_weights=False)[0]
        tgt = q + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        memory_pos=None,
    ):
        tgt2 = self.norm1(tgt)  # todo normalize q only?
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, q):
        # if self.normalize_before:
        #     return self.forward_pre(
        #         tgt, memory_list, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos
        #     )
        return self.forward_post(q)


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(
        self,
        tgt,  # embedding
        cross_input,  # encoder features
    ):
        q = k = v = tgt
        tgt2 = self.self_attn(q, k, v, need_weights=False)[0]
        tgt = q + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(
            query=tgt,
            key=cross_input,
            value=cross_input,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, q, cross_input):
        # if self.normalize_before:
        #     return self.forward_pre(
        #         tgt, memory_list, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, memory_pos
        #     )
        return self.forward_post(q, cross_input)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransDecoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransDecoder, self).__init__()
        attn_layer = CrossAttnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        position_embedding = build_position_encoding(hidden_dim)
        self.pos = position_embedding(torch.ones((1, 36, 36), dtype=torch.bool).cuda())  # torch.Size([1, 128, 36, 36])
        self.pos = self.pos.flatten(2).permute(2, 0, 1)

    def forward(self, imgf, query_embed):
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        # imgf = imgf + self.pos  # add positional encoding  # todo no pos encoding needed here?
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        query_embed = query_embed + self.pos  # add positional encoding

        for layer in self.layers:
            query_embed = layer(query_embed, cross_input=imgf)
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h, w)

        return query_embed


class TransEncoder(nn.Module):
    def __init__(self, num_attn_layers, hidden_dim=128):
        super(TransEncoder, self).__init__()
        attn_layer = SelfAttnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        position_embedding = build_position_encoding(hidden_dim)
        self.pos = position_embedding(torch.ones((1, 36, 36), dtype=torch.bool).cuda())  # torch.Size([1, 128, 36, 36])
        self.pos = self.pos.flatten(2).permute(2, 0, 1)

    def forward(self, imgf):
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)
        imgf = imgf + self.pos  # add positional encoding

        for layer in self.layers:
            imgf = layer(imgf)
        imgf = imgf.permute(1, 2, 0).reshape(bs, c, h, w)

        return imgf


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DocTrOrig(nn.Module):
    def __init__(self, config):
        super(DocTrOrig, self).__init__()
        self.num_attn_layers = config["num_attn_layers"]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        hdim = config["hidden_dim"]

        self.fnet = BasicEncoder(output_dim=hdim, norm_fn="instance")

        self.TransEncoder = TransEncoder(self.num_attn_layers, hidden_dim=hdim)
        self.TransDecoder = TransDecoder(self.num_attn_layers, hidden_dim=hdim)
        self.query_embed = nn.Embedding(1296, hdim)

        self.flow_head = FlowHead(hdim, hidden_dim=hdim)

    def forward(self, image):
        """
        image: segmented image
        """
        fmap = self.fnet(image)
        fmap = torch.relu(fmap)

        fmap = self.TransEncoder(fmap)
        fmap = self.TransDecoder(fmap, self.query_embed.weight)

        dflow = self.flow_head(fmap)

        return {"flow": dflow, "feature_map": fmap}
