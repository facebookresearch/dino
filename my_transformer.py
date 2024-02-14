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
from vit_utils import trunc_normal_
import torch.optim
# from .load_and_shuffle_paths import ImagePathLoader
from load_and_shuffle_paths_perc import ImagePathLoader
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
from tqdm import tqdm


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

class IBVSDataset(Dataset):
    def __init__(self, query_paths, target_paths, velocity_array, device, img_transform=None, vel_transform = None, normalize=False, use_depth=True, blurred=False, random_blur=False):
        self.query_paths = query_paths
        self.target_paths = target_paths
        self.img_transform = img_transform
        self.vel_transform = vel_transform
        self.velocity_array = velocity_array
        self.device = device
        self.normalize = normalize
        self.use_depth = use_depth
        self.learn_vec_len = self.velocity_array[0, :].size
        self.blurred = blurred
        self.random_blur = random_blur

    def __len__(self):
        return len(self.query_paths)

    def is_image_corrupted(self, file_path):
        try:
            Image.open(file_path).verify()
            return False
        except Exception as e:
            rospy.logerr("\033[91m" + f"Error verifying image {file_path}: {e}" + "\033[0m")
            return True

    def __getitem__(self, idx):
        # if self.is_image_corrupted(self.query_paths[idx]):
        #     self.query_paths.pop(idx)
        #     self.target_paths.pop(idx)
        #     self.velocity_array = np.delete(self.velocity_array, idx, axis=0)
        try:
            query_image = Image.open(self.query_paths[idx])
            query_image = copy.deepcopy(query_image)
            target_image = Image.open(self.target_paths[idx])
            target_image = copy.deepcopy(target_image)
        except Exception as e:
            rospy.logerr("\033[91m" + f"Error copying image {self.query_paths[idx]}: {e}" + "\033[0m")
            return torch.zeros([3, 224, 224]).to(self.device), torch.zeros([3, 224, 224]).to(self.device), torch.zeros([1, 1, self.learn_vec_len]).to(self.device)
            
        
        # query_image.show(title="Original")
        if self.blurred and not random.choice([self.random_blur, False]):
            query_image = query_image.filter(ImageFilter.GaussianBlur(radius=3))
            target_image = target_image.filter(ImageFilter.GaussianBlur(radius=3))
            # query_image.show(title="Blurred")

        if self.use_depth:
            query_depth_image = Image.open(self.query_paths[idx].replace('rgb', 'depth'))
            query_depth_image = copy.deepcopy(query_depth_image)
            target_depth_image = Image.open(self.target_paths[idx].replace('rgb', 'depth'))
            target_depth_image = copy.deepcopy(target_depth_image)
        velocity = self.velocity_array[idx, :].reshape((1, self.learn_vec_len))
        if self.img_transform:
            try:
                target_image = self.img_transform(target_image)
                query_image = self.img_transform(query_image)
            except Exception as e:
                rospy.logerr("\033[91m" + f"Error transforming image {self.query_paths[idx]}: {e}" + "\033[0m")
                return torch.zeros([3, 224, 224]).to(self.device), torch.zeros([3, 224, 224]).to(self.device), torch.zeros([1, 1, self.learn_vec_len]).to(self.device)

            if self.use_depth:
                query_depth_image = self.img_transform(query_depth_image)
                target_depth_image = self.img_transform(target_depth_image)
                query_image = torch.cat((query_image, query_depth_image), dim=0)
                target_image = torch.cat((target_image, target_depth_image), dim=0)

            if self.normalize:
                mean, std = target_image.mean([1,2]), target_image.std([1,2])
                norm_tf = transforms.Normalize(mean, std)
                query_image = norm_tf(query_image)
                target_image = norm_tf(target_image)
        
        if self.vel_transform:
            velocity = self.vel_transform(velocity)

        return query_image.to(self.device), target_image.to(self.device), velocity.to(self.device)

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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2): # Added drop
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2): # Added drop
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2): # Added drop
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

def vel_mlp(in_features, hidden_features, out_features, act_layer=nn.ReLU, drop=0.0):
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

    def normalize_velocity_tensor(self, velocity_tensor):
        v_reshaped = velocity_tensor.view(velocity_tensor.shape[0], velocity_tensor.shape[2])
        v_normalized = v_reshaped.clone()
        # v_normalized = self.t_norm.normalize(v_normalized)
        # v_normalized = self.r_norm.normalize(v_normalized)
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
        emb_q = self.vit(query)
        emb_t = self.vit(goal)
        emb_both = torch.cat((emb_t, emb_q), dim=-1)
        if flattened_tensor:
            emb_flatten = emb_both.view(emb_both.shape[0], 1, -1)
            vel = self.mlp(emb_flatten)
        # print(f"{emb_both.shape=}")
        # print(f"{emb_both.view(emb_both.shape[0], 1, -1).shape[-1]=}")
        # vel = self.calculate_velocity_vector(emb_both, output_features, flattened_tensor)
        return vel

class IBVSTransformerTraining():
    def __init__(self, args, pretrained=True, use_loc_tok=True, use_cls_tok=True, flattened_tensor=True, loss_func=nn.SmoothL1Loss(), beta=1, use_depth=False, output_features=[6]):
        self.EPOCHS = args.epochs
        self.BATCH_SIZE = args.batch_size
        self.LEARNING_RATE = args.learning_rate # 2e-4 # 1e-3
        self.weight_decay = args.weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if use_depth:
            in_chans = 4
        else:
            in_chans = 3
        # self.vit = vit_small(pretrained = pretrained, in_chans=in_chans, use_clt=use_cls_tok, use_loc=use_loc_tok).to(self.device)
        if args.activation == 'tanh':
            act_layer = nn.Tanh()
        if args.activation == 'relu':
            act_layer = nn.ReLU()
        token_str = ''
        if use_loc_tok:
            token_str += '_loc'
        if use_cls_tok:
            token_str += '_cls'
        self.token_str = token_str
        self.vit = IBVSTransformer(pretrained=pretrained, in_chans=in_chans, use_loc_tok=use_loc_tok, use_cls_tok=use_cls_tok, output_features=output_features, act_layer=act_layer)
        self.emb_dim = self.vit.vit.embed_dim
        self.output_features = output_features
        """
        self.mlp = None
        self.post_mlp = None
        """
        self.optimizer = torch.optim.AdamW(self.vit.parameters(), self.LEARNING_RATE, weight_decay=self.weight_decay)
        if args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.1, patience=2)
        # self.loss_func = nn.MSELoss()
        self.beta=beta
        self.loss_func = loss_func
        self.loss_func.to(self.device)
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.loss_path = os.path.join(current_path, 'losses')
        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)
        self.model_path = os.path.join(current_path, 'models')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.flattened_tensor=flattened_tensor
        self.use_depth=use_depth

    def calculate_accuracy(self, predictions, targets):
        absolute_difference = torch.linalg.norm(predictions - targets, dim=1)
        targets_norm = torch.linalg.norm(targets, dim=1)
        percentage_difference = absolute_difference / targets_norm
        percentage_difference = torch.clip(percentage_difference, 0, 1)
        accuracy = 1 - torch.mean(percentage_difference)
        return accuracy.item()
    
    def normalize_velocity_tensor(self, velocity_tensor):
        v_reshaped = velocity_tensor.view(velocity_tensor.shape[0], velocity_tensor.shape[2])
        v_normalized = v_reshaped.clone()
        # print(f"calculated: {v_normalized=}")
        v_normalized[:, :3] = nn.functional.normalize(v_reshaped[:, :3], p=2, dim=1)
        if self.learn_vec_len == 8:
            v_normalized[:, 4:7] = nn.functional.normalize(v_reshaped[:, 4:7], p=2, dim=1)
        else:
            v_normalized[:, 3:] = nn.functional.normalize(v_reshaped[:, 3:], p=2, dim=1)
        # print(f"normalized: {v_normalized=}")
        return v_normalized

    def train_transformer(self, args, test_case, it, pretrained=True, 
                          normalize_imgs=False, without_scale=True):
        g = None
        if args.use_seed:
            torch.manual_seed(args.seed)
            g = torch.Generator()
            g.manual_seed(args.seed)
        image_path_loader = ImagePathLoader(from_harddive=True, without_scaling_factors=without_scale)

        train_storer, test_storer = image_path_loader.load_paths_and_velocities_for_training(args.perc_train_traj,
                                                                                             without_scaling_factors=without_scale,
                                                                                             test_stop=True)
        """
        test_iterations = 40
        train_storer_test_target_list = [train_storer.target_list[0]] * (self.BATCH_SIZE*test_iterations)
        train_storer_test_query_list = [train_storer.query_list[0]] * (self.BATCH_SIZE*test_iterations)
        train_storer_test_vel_vec = np.zeros((self.BATCH_SIZE*test_iterations, train_storer.vel_vec.shape[1]))
        train_storer_test_vel_vec[:, :] = train_storer.vel_vec[0, :]
        """
        test = False

        print(f"shuffeler_finished")
        transform = transforms.ToTensor()

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])
        if test:
            train_dataset = IBVSDataset(train_storer_test_query_list, train_storer_test_target_list, train_storer_test_vel_vec,
                                        self.device, img_transform=transform_train, vel_transform=transforms.ToTensor(), normalize=normalize_imgs,
                                        use_depth=self.use_depth, blurred=args.blurred, random_blur=args.random_blur)
            test_dataset = IBVSDataset(train_storer_test_query_list, train_storer_test_target_list, train_storer_test_vel_vec,
                                    self.device, img_transform=transform, vel_transform=transforms.ToTensor(), normalize=normalize_imgs,
                                    use_depth=use_depth, blurred=args.blurred, random_blur=args.random_blur)
        else:
            train_dataset = IBVSDataset(train_storer.query_list, train_storer.target_list, train_storer.vel_vec,
                                        self.device, img_transform=transform_train, vel_transform=transforms.ToTensor(), normalize=normalize_imgs,
                                        use_depth=self.use_depth, blurred=args.blurred, random_blur=args.random_blur)
            test_dataset = IBVSDataset(test_storer.query_list, test_storer.target_list, test_storer.vel_vec,
                                    self.device, img_transform=transform, vel_transform=transforms.ToTensor(), normalize=normalize_imgs,
                                    use_depth=self.use_depth, blurred=args.blurred, random_blur=args.random_blur)
        
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        self.learn_vec_len = train_dataset.learn_vec_len

        total_data = len(train_dataset)
        all_iterations_per_epoch = total_data // self.BATCH_SIZE + (total_data % self.BATCH_SIZE != 0)
        print(f"{all_iterations_per_epoch=}")
        
        it_per_epoch = int(all_iterations_per_epoch * args.perc_train_traj/100)
        print(f"{it_per_epoch=}")

        param_str = f"_lr_{self.LEARNING_RATE}_lossfunc_{self.loss_func.__class__.__name__}_beta_{self.beta}"
        param_str += str(self.weight_decay)
        if normalize_imgs:
            param_str += '_norm'
        if self.flattened_tensor:
            param_str += '_fl'
        param_str += f'_s{self.learn_vec_len}'
        if self.learn_vec_len == 8:
            param_str += '_'
            for of in self.output_features:
                param_str += str(of)
        param_str += self.token_str
        param_str += str(args.batch_size)
        if self.use_depth:
            param_str += '_d'
        if args.blurred:
            param_str += '_bl'
        if args.random_blur:
            param_str += '_rb'
        if args.train_norm and not args.layer_norm:
            param_str += '_tn'
        if args.layer_norm:
            param_str += '_ln'
        param_str += '_' +  args.activation
        if args.use_scheduler:
            param_str += f'_lrs{self.scheduler.__class__.__name__}'
        if args.use_seed:
            param_str += f'_seed{args.seed}'

        train_string = ""
        test_string = ""
        current_lr = args.learning_rate
        train_losses = []
        test_losses = []
        train_accuracies_batches = []
        train_accuracies = []
        test_accuracies_batches = []
        test_accuracies = []
        for epoch in range(self.EPOCHS):
            self.vit.train()
            train_loss_batches = []
            train_loss = 0.0
            train_loss_break = 0.0
            start_ts = time.time()
            dt1 = datetime.fromtimestamp(start_ts)
            # for batch in train_loader:
            with tqdm(train_loader) as tr_loader:
                for i, batch in enumerate(tr_loader, 0):
                    tr_loader.set_description(f"Epoch {epoch + 1}")
                    self.optimizer.zero_grad()
                    query_image, target_image, vel_vec = batch
                    # query_image[2] = torch.zeros([3, 224, 224]) for Test
                    mask = ~torch.all(query_image == 0, dim=(1, 2, 3))
                    query_image = query_image[mask]
                    target_image = target_image[mask]
                    vel_vec = vel_vec[mask]
                    
                    # Finden Sie den Index des ersten Elements, das nur Nullen enthält
                    index_of_zeros_batch = torch.nonzero(mask, as_tuple=True)
                    vel_vec = vel_vec.view(vel_vec.shape[0], 1, self.learn_vec_len)
                    # query_image, target_image, vel_vec = query_image.to(self.device), target_image.to(self.device), vel_vec.to(self.device)
                    """
                    v_normalized = self.calculate_velocity_vector(target_image, query_image, output_features=output_features)
                    """
                    v_normalized = self.vit(goal=target_image,
                                            query=query_image,
                                            output_features=self.output_features,
                                            flattened_tensor=self.flattened_tensor)
                    v_target_normalized = self.normalize_velocity_tensor(vel_vec).to(torch.float32)
                    # print(f"{v_normalized[0]=}\n{v_target_normalized[0]=}")
                    loss = self.loss_func(v_normalized, v_target_normalized).to(torch.float32)
                    train_loss += loss.detach().cpu().item() / len(train_loader)
                    train_loss_break += loss.detach().cpu().item() / it_per_epoch
                    # print(f"{v_normalized.dtype=}_{v_target_normalized.dtype=}_{loss.dtype=}")
                    loss.backward()
                    self.optimizer.step()

                    train_loss_batches.append(loss.item())
                    accuracy = self.calculate_accuracy(v_normalized, v_target_normalized)
                    train_accuracies_batches.append(accuracy)
                    tr_loader.set_postfix(loss=loss.item(), accuracy=accuracy, lr=current_lr)
                    
                    # print(f"epoch: {epoch + 1}, valid_in_batch: {query_image.shape[0]}, loss: {loss.detach().cpu().item()}, accuracy: {accuracy}")
                    
                    if args.use_given_its and i == it_per_epoch:
                        break
            print(f"Total iterations {i} for epoch {epoch}")
            # print(f"{loss.detach().cpu().item()=}_{v_normalized=}_{v_target_normalized=}")
            end_ts = time.time()
            dt2 = datetime.fromtimestamp(end_ts)
            # Difference between two timestamps
            # in hours:minutes:seconds format
            delta = dt2 - dt1
            loss_train = np.mean(train_loss_batches)
            self.scheduler.step(loss_train)
            current_lr = self.scheduler.get_last_lr()[0]
            # Learning rate adaption
            accuracies_train = np.mean(train_accuracies_batches)
            train_accuracies.append(accuracies_train)
            train_losses.append(loss_train)
            train_string = train_string + f"Epoch {epoch + 1}/{self.EPOCHS}, time: {delta}, loss_for_loader_len: {train_loss:.4f}, loss: {loss_train:.4f}, loss_type: {self.loss_func.__class__.__name__}, accuracy: {accuracies_train:.4f}" + "\n"
            print(f"Epoch {epoch + 1}/{self.EPOCHS}, time: {delta}, loss_for_loader_len: {train_loss:.4f}, loss: {train_loss_break:.4f}")
            model_file = f'{self.model_path}/ibvs_vit_pretrained_{pretrained}_case_{test_case}_it_{it_per_epoch}_epoch_{epoch + 1}' + param_str
            if args.random_blur:
                model_file += '_rb'
            if test:
                model_file += '_test'
            torch.save(self.vit.state_dict(), model_file + '.pth')

            # Test loop
            self.vit.eval()
            with torch.no_grad():
                test_loss_batches = []
                test_loss = 0.0
                # for batch in test_loader:
                for i, batch in enumerate(test_loader, 0):
                    query_image, target_image, vel_vec = batch
                    vel_vec = vel_vec.view(vel_vec.shape[0], 1, self.learn_vec_len)
                    # query_image, target_image, vel_vec = query_image.to(self.device), target_image.to(self.device), vel_vec.to(self.device)
                    v_normalized = self.vit(goal=target_image,
                                            query=query_image,
                                            output_features=self.output_features,
                                            flattened_tensor=self.flattened_tensor)
                    v_target_normalized = self.normalize_velocity_tensor(vel_vec)
                    loss = self.loss_func(v_normalized, v_target_normalized).to(torch.float)
                    test_loss += loss.detach().cpu().item() / len(test_loader)
                    test_loss_batches.append(loss.item())
                    test_accuracy = self.calculate_accuracy(v_normalized, v_target_normalized)
                    test_accuracies_batches.append(test_accuracy)
                loss_test = np.mean(test_loss_batches)
                test_losses.append(loss_test)
                test_accuracy = np.mean(test_accuracies_batches)
                test_accuracies.append(test_accuracy)

                if test:
                    test_string += 'Test'
                test_string = test_string + f"Epoch {epoch + 1}/{self.EPOCHS}, loss: {test_loss:.4f}" + "\n"
                print(f"Test loss: {test_loss:.4f}")
                
        
        with open(f"{self.loss_path}/testloss_{test_case}_pt_{pretrained}_{test_case}_it_{it_per_epoch}" + param_str, 'w') as file:
            # dump information to that file
            file.write(test_string)

        with open(f"{self.loss_path}/trainloss_{test_case}_pt_{pretrained}_{test_case}_it_{it_per_epoch}" + param_str, 'w') as file:
            # dump information to that file
            file.write(train_string)
        
        """
        test_string = ""
        # Test loop
        self.vit.eval()
        with torch.no_grad():
            valid_loss_batches = []
            test_loss = 0.0
            # for batch in test_loader:
            for i, data in enumerate(test_loader, 0):
                query_image, target_image, vel_vec = batch
                vel_vec = vel_vec.view(vel_vec.shape[0], 1, self.learn_vec_len)
                # query_image, target_image, vel_vec = query_image.to(self.device), target_image.to(self.device), vel_vec.to(self.device)
                v_normalized = self.calculate_velocity_vector(target_image, query_image, output_features=output_features)
                v_target_normalized = self.normalize_velocity_tensor(vel_vec)
                loss = self.loss_func(v_normalized, v_target_normalized)
                test_loss += loss.detach().cpu().item() / len(test_loader)
                valid_loss_batches.append(loss.item())
            loss_val = np.mean(valid_loss_batches)
            valid_loss.append(loss_val)

            test_string = test_string + f"loss: {test_loss:.4f}" + "\n"
            print(f"Test loss: {test_loss:.4f}")
        with open(f"{self.loss_path}/loss_{pretrained}_{test_case}_test_{it_per_epoch}_lr_{self.LEARNING_RATE}_lossfunc_{self.loss_func}_n_{normalize_imgs}_f_{self.flattened_tensor}", 'w') as file:
            # dump information to that file
            file.write(test_string)
        """

        # visualize the loss as the network trained
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
        ax1.plot(range(1,len(test_losses)+1),test_losses,label='Test Loss')
        # find position of lowest validation loss
        minposs = test_losses.index(min(test_losses))+1 
        ax1.axvline(minposs, linestyle='--', color='r',label='Lowest test loss')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.grid(True)
        ax1.legend()
        
        ax1.set_title('Loss')

        ax2.plot(range(1,len(train_accuracies)+1),train_accuracies, label='Training Accuracy')
        ax2.plot(range(1,len(test_accuracies)+1),test_accuracies,label='Test Accuracy')
        # find position of lowest validation loss
        maxposs = test_accuracies.index(max(test_accuracies))+1
        ax2.axvline(maxposs, linestyle='--', color='r',label='Highest test accuracy')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Accuracy')

        plt.tight_layout()
        fig.savefig('combined_plot.png', bbox_inches='tight')
        img_file = f'{self.model_path}/ibvs_vit_pretrained_{test_case}_pt_{pretrained}_it_{it_per_epoch}' + param_str
        if test:
            img_file + '_test'
        plt.savefig(img_file + '.png')
        """
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
        plt.plot(range(1,len(test_losses)+1),valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(test_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Lowest validation loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        # plt.ylim(0, 3) # consistent scale
        # plt.xlim(1, len(train_losses)) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig('loss_plot.png', bbox_inches='tight')
        img_file = f'{self.model_path}/losses_ibvs_vit_pretrained_{pretrained}_it_{it_per_epoch}_lr_{self.LEARNING_RATE}_lossfunc_{self.loss_func}_beta_{self.beta}_n_{normalize_imgs}_f_{self.flattened_tensor}_ws_{without_scale}_depth_{use_depth}'
        if test:
            img_file + '_test'
        plt.savefig(img_file + '.png')
        """
        
        return self.vit
        """
        for epoch in self.EPOCHS:
            self.vit.train()
            for idx in range(1, len(train_target_paths)):
                self.optimizer.zero_grad()
                query_image = Image.open(train_query_paths[idx - 1])
                target_image = Image.open(train_target_paths[idx - 1])
                v_normalized = self.calculate_velocity_vector(query_image, target_image)
                loss = self.loss_func(v_normalized)
                loss.backward()
                self.optimizer.step()
        torch.save(self.vit.state_dict(), 'ibvs_vit.pth')
        return self.vit
        """
if __name__ == '__main__':
    trans_trainer = IBVSTransformerTraining(pretrained=True)
    trans_trainer.train_transformer()