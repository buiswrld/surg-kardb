#Base Imports 
import math
from functools import partial
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
import collections.abc
import math
import re
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, Iterator, Tuple, Type, Union
import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint

#GCN Imports
import os
import urllib.request
from urllib.error import HTTPError
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.nn import global_mean_pool
gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

""" 
Implementation Breakdown:
1. Create and install conda env to run the following command:
 - Create a weights and biases account and enter login within terminal after pip install 
 - cliff_base main.py train --dataset_path /pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation/action_dataset_joints_leg_sampled_50.pkl --seq_len 50 --embedd_dim 84 --batch_size 32 --exp_name abl_seq_pelvis_arm_head_thorax_spine_leg_weighted_seq_50
2. Run the below sandbox function and set breakpoints at the __init__ and _forward__ functions in the GNNModel class to inspect dimensions 
3. [ASYNC] Abstraction of the .pkl reference to matrices of nodes and edge matrices 
    - Input PKL > Output Matrices
    - *can be run in a separate sandbox*
    - Edge matrix creation logic (using pre-mapped indices for connected joints) and *MUST* account for 1) temporal and 2) spatial connections 
    - Generate pre-mapped indices automatically based on joint selections (i.e. remapping conections if we exclude the spine/thorax)
4. [ASYNC] Dataloader abstraction and adaptation to torch_geometric formats 
    - Input Matrices from (3) > Output PyTorch Geometric Dataset 
    - *can be run in a separate sandbox* 
    = Note: Need to load in (number of nodes, number of features) and an additional ID matrix that maps each node to which "graph" it belongs to 
5. [ASYNC] Hyperparam tuning/initial experiments with the full pipeline 
    - Starting Params
        - c_in = 3
        - c_hidden = 128 
        - c_out = 3 
        - num_layers = 5 
    - Tuning Params:
        - c_hidden
        - num_layers 
        - [Meta] adjacency contruction
            - weights
            - number of connections
            - temporal vs. spatial connections
"""

class GNNModel(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_hidden=16,
        c_out=3,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)

        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        #breakpoint() 
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch = None):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        """

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            #breakpoint() 
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x, edge_index)
                #breakpoint() 
            else:
                x = layer(x)
                #breakpoint() 
        x = global_mean_pool(x, batch) if batch is not None else x
        return x

def gnn_sandbox_function():
    model = GNNModel() 
    input = torch.rand((28, 3))
    edge_matrix = torch.tensor([
        [0, 1, 1, 2],  # Source nodes
        [1, 0, 2, 1]   # Target nodes
    ])
    output = model(input) 
    #breakpoint() 

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
        ) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            hmr_embedd_dim,
            seq_len=145, #Previously number patches 
            mlp_ratio=(0.5, 4.0), #Only determines hidden states 
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * hmr_embedd_dim) for x in to_2tuple(mlp_ratio)] #Tokens_dim and channels_dim are both hidden states (256, 2048)
        self.tokens_dim = tokens_dim
        self.channels_dim = channels_dim
        self.norm1 = norm_layer(hmr_embedd_dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop) #145 => 256 => 145   
        self.mlp_channels = mlp_layer(hmr_embedd_dim, channels_dim, act_layer=act_layer, drop=drop) #512 => 2048 => 512 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(hmr_embedd_dim)

    def forward(self, x): 
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)
    

class MlpMixer(nn.Module):
    def __init__(
            self,
            num_classes=3,
            num_blocks=8,
            hmr_embedd_dim=512, #TODO -- should customize 
            seq_len=145, #TODO -- should customize
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            global_pool='avg',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = hmr_embedd_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.blocks = nn.Sequential(*[
            block_layer(
                hmr_embedd_dim,
                seq_len,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(num_blocks)])
        self.norm = norm_layer(hmr_embedd_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(hmr_embedd_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        #breakpoint() 
        x = self.forward_features(x) # torch.Size([32, 50, 84]) (batch_size, seq_len, embedd_dim)
        x = self.forward_head(x) # torch.Size([32, 3]) 
        #breakpoint() 
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

if __name__ == "__main__":
    print("Starting") 
    model = MlpMixer()
    input = torch.rand((1, 145, 512))
    out = model(input)
    #breakpoint() 