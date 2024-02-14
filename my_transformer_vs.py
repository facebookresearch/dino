# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Mostly copy-paste from timm library.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
from functools import partial

import torch
import torch.nn as nn
import os
from .vit_utils import trunc_normal_
import torch.optim
from .load_and_shuffle_paths import ImagePathLoader
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import time
from datetime import datetime
import matplotlib.pyplot as plt
import rospy
import random
import copy


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# use_clt .. use classification token
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_clt = True, use_loc = True, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.use_classification_token = use_clt
        self.use_localization_tokens = use_loc

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_localization_tokens and self.use_classification_token:
            return x
        elif self.use_localization_tokens:
            return x[:, 1:]
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, use_clt = True, use_loc = True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_clt=use_clt, use_loc=use_loc, **kwargs)
    return model

"""
def vit_small(patch_size=16, pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load('facebookresearch/dino:main', 'dino_vits16').state_dict(), strict=False)
    return model
"""
def vit_small(patch_size=16, in_chans=3, use_avg_pooling_and_fc=True, num_classes=128, pretrained=True, use_clt = True, use_loc = True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, in_chans=in_chans, use_avg_pooling_and_fc=use_avg_pooling_and_fc, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_clt=use_clt, use_loc=use_loc, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load('facebookresearch/dino:main', 'dino_vits16').state_dict(), strict=False)
    return model

def vit_base(patch_size=16, use_clt = True, use_loc = True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_clt=use_clt, use_loc=use_loc, **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        return x

"""
MLP for complete velocity and scale vector
"""
class NormalizationLayer(nn.Module):
    def __init__(self, start_idx, end_idx):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        
    def forward(self, x):
        y = x.clone()
        y[:, self.start_idx:self.end_idx] = nn.functional.normalize(x[:, self.start_idx:self.end_idx], p=2, dim=1)
        return y

class Mlp_all(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.2): # Added drop
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features[0])
        self.drop = nn.Dropout(drop)
        self.t_norm = NormalizationLayer(start_idx=0, end_idx=3)
        start_idx = int(sum(out_features)/2)
        end_idx = start_idx + 3
        # self.r_norm = Norm(start_idx=start_idx, end_idx=end_idx)
        self.r_norm = NormalizationLayer(start_idx=start_idx, end_idx=end_idx)
        self.id_act = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x_norm = x.clone().view(x.shape[0], x.shape[2])
        x_norm = self.t_norm(x_norm)
        x_norm = self.r_norm(x_norm)
        x_norm = self.id_act(x_norm)
        return x_norm

class Mlp_tr(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.2): # Added drop
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3t = nn.Linear(hidden_features, out_features[0])
        self.fc3r = nn.Linear(hidden_features, out_features[1])
        self.drop = nn.Dropout(drop)
        self.norm = NormalizationLayer(start_idx=0, end_idx=3)
        self.id_act = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        t = self.fc3t(x)
        t_norm = t.clone().view(t.shape[0], t.shape[2])
        t_norm = self.norm(t_norm)
        r = self.fc3r(x)
        r_norm = r.clone().view(r.shape[0], r.shape[2])
        r_norm = self.norm(r_norm)
        concat_list = [t_norm, r_norm]
        y = torch.cat(concat_list, dim=-1)
        y = self.id_act(y)
        return y

class Mlp_tsrs(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.2): # Added drop
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3t = nn.Linear(hidden_features, out_features[0])
        self.fc3ts = nn.Linear(hidden_features, out_features[1])
        self.fc3r = nn.Linear(hidden_features, out_features[2])
        self.fc3rs = nn.Linear(hidden_features, out_features[3])
        self.drop = nn.Dropout(drop)
        self.norm = NormalizationLayer(start_idx=0, end_idx=3)
        self.id_act = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        t = self.fc3t(x)
        t_norm = t.clone().view(t.shape[0], t.shape[2])
        t_norm = self.norm(t_norm)
        s_t = self.fc3ts(x)
        s_t = s_t.clone().view(s_t.shape[0], s_t.shape[2])
        t_concat_list = [t_norm, s_t]
        y_t = torch.cat(t_concat_list, dim=-1)
        r = self.fc3r(x)
        r_norm = r.clone().view(r.shape[0], r.shape[2])
        r_norm = self.norm(r_norm)
        s_r = self.fc3rs(x)
        s_r = s_r.clone().view(s_r.shape[0], s_r.shape[2])
        r_concat_list = [r_norm, s_r]
        y_r = torch.cat(r_concat_list, dim=-1)
        concat_list = [y_t, y_r]
        y = torch.cat(concat_list, dim=-1)
        y = self.id_act(y)
        return y

def vel_mlp(in_features, hidden_features, out_features, act_layer=nn.ReLU(), drop=0.0):
    if len(out_features)==1:
        model = Mlp_all(in_features=in_features,
                        hidden_features=hidden_features,
                        out_features=out_features,
                        act_layer=act_layer)
    elif len(out_features)==2:
        model = Mlp_tr(in_features=in_features,
                       hidden_features=hidden_features,
                       out_features=out_features,
                       act_layer=act_layer)
    elif len(out_features)==4:
        model = Mlp_tsrs(in_features=in_features,
                         hidden_features=hidden_features,
                         out_features=out_features,
                         act_layer=act_layer)
    return model


class IBVSTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,pretrained=True, in_chans=3, use_loc_tok=True, use_cls_tok=True, output_features=[6], act_layer=nn.Tanh()):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit = vit_small(pretrained=pretrained, in_chans=in_chans, use_clt=use_cls_tok, use_loc=use_loc_tok).to(self.device)
        self.emb_dim = self.vit.embed_dim
        token_size = 0
        if use_loc_tok:
            token_size += 196
        if use_cls_tok:
            token_size += 1
        in_features = token_size * 2 * self.emb_dim
        self.mlp = vel_mlp(in_features=in_features,
                          hidden_features=4*self.emb_dim,
                          out_features=output_features,
                          act_layer=act_layer,
                          drop=0.2).to(self.device)
        self.post_mlp = None
        # self.t_norm = Norm(start_idx=0, end_idx=3)
        self.t_norm = NormalizationLayer(start_idx=0, end_idx=3).to(self.device)
        start_idx = int(sum(output_features)/2)
        end_idx = start_idx + 3
        # self.r_norm = Norm(start_idx=start_idx, end_idx=end_idx)
        self.r_norm = NormalizationLayer(start_idx=start_idx, end_idx=end_idx).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def normalize_velocity_tensor(self, velocity_tensor, transform=transforms.ToTensor()):
        v_reshaped = transform(velocity_tensor.reshape((len(velocity_tensor), 1)))
        v_normalized = v_reshaped.clone()
        v_normalized = self.t_norm(v_normalized)
        v_normalized = self.r_norm(v_normalized)
        return v_normalized

    def calculate_velocity_vector(self, emb_both, output_features=None, flattened_tensor = True):
        # emb_both = torch.cat((emb_q, emb_t), dim=1)
        if flattened_tensor:
            emb_flatten = emb_both.view(emb_both.shape[0], 1, -1)
            # embed_dim_1 = emb_flatten.shape[1]
            v = self.mlp(emb_flatten)
        else:
            if self.mlp is None:
                self.mlp = nn.Sequential(
                    nn.Linear(self.emb_dim, 4*self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(4*self.emb_dim, 4*self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(4*self.emb_dim, self.learn_vec_len),
                    nn.ReLU()).to(self.device)
            v_patches = self.mlp(emb_both)
            v_patches_trans = v_patches.permute(0, 2, 1)
            if self.post_mlp is None:
                post_vit_dim = v_patches_trans.shape[-1]
                self.post_mlp = nn.Sequential(
                    nn.Linear(post_vit_dim, 4*self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(4*self.emb_dim, 4*self.emb_dim),
                    nn.ReLU(),
                    nn.Linear(4*self.emb_dim, 1),
                    nn.ReLU()).to(self.device)
            v = self.post_mlp(v_patches_trans).permute(0, 2, 1)
        v_normalized = v.view(v.shape[0], v.shape[2])
        # v_normalized = v.squeeze(1)
        v_normalized = self.normalize_velocity_tensor(v)
        # v_normalized = v
        return v_normalized

    def forward(self, goal, query, output_features, flattened_tensor):
        emb_q = self.vit(self.transform(query).unsqueeze(0).to(self.device))
        emb_t = self.vit(self.transform(goal).unsqueeze(0).to(self.device))
        emb_both = torch.cat((emb_t, emb_q), dim=-1)
        if flattened_tensor:
            emb_flatten = emb_both.view(emb_both.shape[0], 1, -1)
            vel = self.mlp(emb_flatten)
        return vel

def ibvs_vit(load_state_dict, model_weights_path, output_features = None, pretrained=True, in_chans=3, use_loc_tok=True, use_cls_tok=True, act_layer=nn.Tanh()):
    model = IBVSTransformer(pretrained=pretrained, output_features = output_features, in_chans=in_chans, use_loc_tok=use_loc_tok, use_cls_tok=use_cls_tok, act_layer=act_layer)
    if load_state_dict:
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()
    return model