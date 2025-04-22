""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .CTrans_F import *
from collections import OrderedDict
from torch_geometric.nn import SAGEConv,LayerNorm

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from .pretrainmodel import SAINT
from .unetr import *


class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        out = out * x
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn


    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            )


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))




def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = x.shape[0] if size is None else size

        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        for i in range(x.shape[0]):
            gate[i] = softmax(gate[i], batch[i].squeeze(-1),num_nodes=1)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)



class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (B, C, H, W, D)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (B, C, H, W, D)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention * x

    

class UNETR_clinical(nn.Module):
    def __init__(self, img_shape=(256, 256, 16), input_dim=4, output_dim=1, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1, cat_dims= None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )
        

        self.ca = ChannelAttention3D(embed_dim*2)
        self.sa = SpatialAttention3D()


        self.classifier_censorship = nn.Linear(embed_dim*2, 1).cuda()
        self.Risk_evalutor = nn.Linear(embed_dim*2, 1).cuda()
        self.saint_process = SAINT(categories = tuple(cat_dims),
                                num_continuous =1)



    def forward(self,x,x_categ_enc, x_cont_enc):
        z = self.transformer(x)
        x_cli = self.saint_process.transformer(x_categ_enc, x_cont_enc)

        z3, z6, z9, z12 = z
        img_feature = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        b,c,_ = img_feature.shape

        x_cli = x_cli.permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        mix_feature = torch.cat((img_feature,x_cli[:,:,0,:,:]),dim=1)
        mix_feature = self.ca(mix_feature)
        mix_feature = self.sa(mix_feature)



        mix_feature = mix_feature.view(b,-1)
        cls_output = self.classifier_censorship(mix_feature)

        risk = self.Risk_evalutor(mix_feature)

        return nn.Sigmoid()(cls_output),nn.Sigmoid()(risk)
    
