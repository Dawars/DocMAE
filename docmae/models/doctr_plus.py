"""
Implementation of DocTr++ based on DocTr code
MIT Licence https://github.com/fh2019ustc/DocTr
"""
import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from docmae.models.transformer import build_position_encoding


class SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        extra_attention=False,
    ):
        super().__init__()
        self.extra_attention = extra_attention
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

    def forward_post(self, query, key, value, cross_kv=None, pos_list=[None, None, None]):
        q = self.with_pos_embed(query, pos_list[0])
        k = self.with_pos_embed(key, pos_list[1])
        v = self.with_pos_embed(value, pos_list[2])

        tgt2 = self.self_attn(q, k, v, need_weights=False)[0]
        tgt = query + self.dropout1(tgt2)  # query and key should be equal, using it for skip connection
        tgt = self.norm1(tgt)

        if self.extra_attention:
            if cross_kv is not None:
                q = tgt
                k = v = cross_kv
            else:
                q = k = v = tgt
            q = self.with_pos_embed(q, pos_list[0])
            k = self.with_pos_embed(k, pos_list[1])
            v = self.with_pos_embed(v, pos_list[2])

            tgt2 = self.cross_attn(q, k, v, need_weights=False)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, query, key, value, cross_kv, pos_list):
        return self.forward_post(query, key, value, cross_kv=cross_kv, pos_list=pos_list)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos


class CrossAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
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

    def forward_post(
        self,
        query,
        key,
        value,
        cross_kv,
        pos_list=[None, None, None]
        # query embedding  # encoder features  # positional encoding
    ):
        q = self.with_pos_embed(query, pos_list[0])
        k = self.with_pos_embed(key, pos_list[1])
        v = self.with_pos_embed(value, pos_list[2])

        tgt2 = self.self_attn(q, k, v, need_weights=False)[0]
        tgt = q + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        q = tgt
        k, v = cross_kv
        q = self.with_pos_embed(q, pos_list[0])
        k = self.with_pos_embed(k, pos_list[1])
        v = self.with_pos_embed(v, pos_list[2])

        tgt2 = self.cross_attn(q, k, v, need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, query, key, value, cross_kv, pos_list):
        return self.forward_post(query, key, value, cross_kv, pos_list=pos_list)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos


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
    def __init__(self, num_attn_layers, hidden_dim=128, pos_encoding_before=True, pos_encoding_value=True):
        super(TransDecoder, self).__init__()
        self.pos_encoding_before = pos_encoding_before
        self.pos_encoding_value = pos_encoding_value
        attn_layer = CrossAttnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        position_embedding = build_position_encoding(hidden_dim)
        self.pos = [
            position_embedding(torch.ones((1, 36 // 2**i, 36 // 2**i), dtype=torch.bool, device="cuda"))
            .flatten(2)
            .permute(2, 0, 1)
            for i in reversed(range(3))
        ]  # torch.Size([1, 128, 36 / 2^i, 36])

    def forward(self, imgf_list, query_embed):
        bs, c, h, w = imgf_list[0].shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        for i, layer in enumerate(self.layers):
            if i in [0, 2, 4]:
                imgf = imgf_list[i // 2].flatten(2).permute(2, 0, 1)
                cross_k = cross_v = imgf

            query = key = value = query_embed

            pos = self.pos[i // 2]
            if self.pos_encoding_before:
                pos_list = [None] * 3
                if i == 0:
                    query = query + pos
                    key = key + pos
                    # cross_k = cross_k + self.pos  # in orig transformer PE is not added here at all
                    if self.pos_encoding_value:
                        value = value + pos
                        # cross_v = cross_v + self.pos
            else:  # add PE every block, also added to cross k,v
                if self.pos_encoding_value:
                    pos_list = [pos] * 3
                else:
                    pos_list = [pos, pos, None]
            query_embed = layer(query=query, key=key, value=value, cross_kv=[cross_k, cross_v], pos_list=pos_list)

            # the decoded embeddings of the first and second blocks are upsampled based on the bilinear interpolation
            if i in [1, 3]:
                query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h * 2 ** (i // 2), w * 2 ** (i // 2))
                query_embed = nn.functional.interpolate(query_embed, scale_factor=2, mode="bilinear", align_corners=True)
                query_embed = query_embed.flatten(2).permute(2, 0, 1)

        scale_factor = 2 ** ((len(self.layers) - 1) // 2)  # 4
        query_embed = query_embed.permute(1, 2, 0).reshape(bs, c, h * scale_factor, w * scale_factor)

        return query_embed


class TransEncoder(nn.Module):
    def __init__(
        self,
        num_attn_layers,
        hidden_dim=128,
        extra_attention=False,
        extra_skip=False,
        pos_encoding_before=True,
        pos_encoding_value=True,
    ):
        super(TransEncoder, self).__init__()
        if not extra_attention:
            assert not extra_skip
        self.extra_skip = extra_skip
        self.pos_encoding_before = pos_encoding_before
        self.pos_encoding_value = pos_encoding_value
        attn_layer = SelfAttnLayer(hidden_dim, extra_attention=extra_attention)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.conv_stride = nn.ModuleList(  # todo what kernel size? depth wise separable?
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
                for _ in range(num_attn_layers // 2 - 1)
            ]
        )
        position_embedding = build_position_encoding(hidden_dim)
        # run here because sin PE is not learned
        self.pos = [
            position_embedding(torch.ones((1, 36 // 2**i, 36 // 2**i), dtype=torch.bool, device="cuda"))
            .flatten(2)
            .permute(2, 0, 1)
            for i in range(3)
        ]  # torch.Size([1, 128, 36 / 2^i, 36])

    def forward(self, imgf):
        bs, c, h, w = imgf.shape
        imgf = imgf.flatten(2).permute(2, 0, 1)

        outputs = []

        for i, layer in enumerate(self.layers):
            query = key = value = imgf
            pos = self.pos[i // 2]
            if self.pos_encoding_before:
                pos_list = [None] * 3
                if i == 0:
                    query = query + pos
                    key = key + pos
                    if self.pos_encoding_value:
                        value = value + pos
            else:  # add PE every block
                if self.pos_encoding_value:
                    pos_list = [pos] * 3
                else:
                    pos_list = [pos, pos, None]
            cross_kv = query if self.extra_skip else None  # extra skip connection from block input before blockwise PE
            imgf = layer(query=query, key=key, value=value, cross_kv=cross_kv, pos_list=pos_list)
            if i in [1, 3, 5]:
                imgf = imgf.permute(1, 2, 0).reshape(bs, c, h // 2 ** (i // 2), w // 2 ** (i // 2))
                outputs.append(imgf)  # save output

                # downsampling after first and second block
                if i // 2 < len(self.conv_stride):
                    imgf = self.conv_stride[i // 2](imgf)
                    imgf = imgf.flatten(2).permute(2, 0, 1)

        return list(reversed(outputs))


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DocTrPlus(nn.Module):
    def __init__(self, config):
        super(DocTrPlus, self).__init__()
        self.num_attn_layers = config["num_attn_layers"]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        hdim = config["hidden_dim"]

        self.trans_encoder = TransEncoder(
            self.num_attn_layers,
            hidden_dim=hdim,
            extra_attention=config["extra_attention"],  # corresponds to cross attention block in decoder
            extra_skip=config["extra_skip"],  # k,v comes from block input (q)
            pos_encoding_before=not config["add_pe_every_block"],  # only add PE once before encoder blocks
            pos_encoding_value=not config["no_pe_for_value"],
        )
        self.trans_decoder = TransDecoder(
            self.num_attn_layers,
            hidden_dim=hdim,
            pos_encoding_before=not config["add_pe_every_block"],
            pos_encoding_value=not config["no_pe_for_value"],
        )
        self.query_embed = nn.Embedding(9 * 9, hdim)  # (288 / 32) ^2

        self.flow_head = FlowHead(hdim, hidden_dim=hdim)

    def forward(self, backbone_features):
        """
        image: segmented image
        """
        fmap = torch.relu(backbone_features)

        fmap = self.trans_encoder(fmap)
        fmap = self.trans_decoder(fmap, self.query_embed.weight)

        dflow = self.flow_head(fmap)

        return {"flow": dflow, "feature_map": fmap}
