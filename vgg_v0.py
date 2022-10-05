import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import resnet26d, resnet50d
from timm.models.registry import register_model
import numpy as np
import math
from timm.models.helpers import build_model_with_cfg

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth')}

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #weights = attn if self.vis else None
        weights = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformerDistilled(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.mask = Mask(384, 4)
        self.part = Mask(384, 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.unflatten = nn.Unflatten(2, (16, 16))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        # self.tgt_f = nn.Parameter(torch.ones(1, 768))
        # self.tgt_b = nn.Parameter(torch.zeros(1, 768))
        # self.tgt = torch.stack([self.tgt_f, self.tgt_b], dim=1).cuda()
        self.tgt = nn.Parameter(torch.randn(1, 2, 384))
        self.tgt_p = nn.Parameter(torch.zeros(1, 4, 384))
        # self.num_prefix_tokens = 2
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        # self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        # self.distilled_training = False  # must set this True to train w/ distillation token

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x):
        x, features, attn_weights = self.forward_features(x)  # features.size(): 2, 769, 196
        tgt = self.tgt.repeat(x.size(0), 1, 1)
        tgt_p = self.tgt_p.repeat(x.size(0), 1, 1)
        outs, masks = self.mask(features, tgt)
        mask_x1, mask_x2 = masks.split(split_size=1, dim=1)
        n, p, c = mask_x1.shape
        out_1, out_2 = outs.split(split_size=1, dim=1)
        out1, _ = self.part(tgt_p, out_1)
        out2, _ = self.part(tgt_p, out_2)
        mask_x1 = torch.reshape(mask_x1, [n, 1, 14, 14])
        mask_x1 = mask_x1.contiguous()
        mask_x2 = torch.reshape(mask_x2, [n, 1, 14, 14])
        mask_x2 = mask_x2.contiguous()
        mask = torch.cat([mask_x1, mask_x2], dim=1)
        out_x = self.head(out1.squeeze(1))
        out_y = self.head(out2.squeeze(1))
        if self.training:
            return out_x,  out_y,  mask
        else:
            # attn_weights = torch.stack(attn_weights)
            # attn_weights = torch.mean(attn_weights, dim=2)
            # # print(attn_weights.size())
            # cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, 14, 14]).unsqueeze(1)
            # # print(cams.size())
            # mask_x1 = mask_x1 * cams
            return out_x,  mask_x1.squeeze(1)

    def get_loss(self, logits, gt_labels):
        gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits, gt)
        return loss_cls

    def get_bceloss(self, logits, gt_labels):
        gt = gt_labels.float()
        loss_cls = self.loss_bce(logits, gt)
        return loss_cls

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def model(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_deit(
        'deit_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **model_kwargs)
    return model

def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')
    model_cls = VisionTransformerDistilled
    model = build_model_with_cfg(
        model_cls, variant, pretrained, default_cfgs,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        **kwargs)
    return model


class Mask(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.normalize_before = normalize_before

    def forward(self, memory, tgt):
        tgt2, masks = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, masks

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=False,
                 activation=F.relu):
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        # self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        Q_split = q.split(split_size=self.in_features // self.head_num, dim=2)
        K_split = k.split(split_size=self.in_features // self.head_num, dim=2)
        V_split = v.split(split_size=self.in_features // self.head_num, dim=2)
        outs = []
        masks = 0
        for i in range(len(Q_split)):
            mask, out = ScaledDotProductAttention()(Q_split[i], K_split[i], V_split[i])
            masks = masks + mask
            outs.append(out)
        masks = masks/self.head_num
        outs = torch.cat(outs, dim=2)
        return outs, masks

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        attention = F.softmax(scores, dim=1)
        return attention, attention.matmul(value)

def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict

def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict.
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

if __name__ == '__main__':

    a = np.random.rand(16, 28, 28)
    b = np.flip(a, 1)
    c = np.flip(a, 2)
    d = 1