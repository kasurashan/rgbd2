# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
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

# from net_utils_ours import Fusionmodel, Addmodel   # fusion module
# from net_utils_intra import Fusionmodel, Addmodel   # fusion module
from net_utils_inter import Fusionmodel, Addmodel   # fusion module

import functools
print = functools.partial(print, flush=True)

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    # def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
    def __init__(self, backbone: nn.Module, backbone_d: nn.Module, train_backbone: bool, return_interm_layers: bool):  ###
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:   # default가 true 라고 보면 될 듯 (default : args.num_features = 4 > 1 True)
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}  
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.body_d = IntermediateLayerGetter(backbone_d, return_layers=return_layers) ##  
        # self.FusionBlock_0 = Addmodel(in_channels=256) ##
        # self.FusionBlock_1 = Fusionmodel(in_channels=512) ##
        # self.FusionBlock_2 = Fusionmodel(in_channels=1024) ##
        # self.FusionBlock_3 = Fusionmodel(in_channels=2048) ##
        self.FusionBlock_0 = Addmodel(in_channels=256)
        self.FusionBlock_1 = Fusionmodel(in_channels=512, patch_h=6, patch_w=6)
        self.FusionBlock_2 = Fusionmodel(in_channels=1024, patch_h=6, patch_w=6)
        self.FusionBlock_3 = Fusionmodel(in_channels=2048, patch_h=6, patch_w=6)

    def forward(self, tensor_list: NestedTensor):
        # xs = self.body(tensor_list.tensors)
        xs = OrderedDict()   #output (fused)
        xs_original = OrderedDict()   # only rgb output
        
        x_rgb = tensor_list.tensors[:,0:3,:,:]
        x_d = tensor_list.tensors[:,3:,:,:]
        x_d = torch.cat((x_d,x_d,x_d), dim=1)

        x_rgb = self.body['conv1'](x_rgb)
        x_rgb = self.body['bn1'](x_rgb)
        x_rgb = self.body['relu'](x_rgb)
        x_rgb = self.body['maxpool'](x_rgb)
        x_rgb = self.body['layer1'](x_rgb)   # [2,256,h/4,w/4]
        xs_original['0'] = x_rgb
        x_d = self.body_d['conv1'](x_d)
        x_d = self.body_d['bn1'](x_d)
        x_d = self.body_d['relu'](x_d)
        x_d = self.body_d['maxpool'](x_d)
        x_d = self.body_d['layer1'](x_d)   # [2,256,h/4,w/4]
        # x_rgb, x_d, x_fused = self.FusionBlock_0(x_rgb, x_d)
        #xs['0'] = x_fused ### solq는 2번째 layer 아웃풋부터 쓰는듯

        x_rgb = self.body['layer2'](x_rgb)   # [2, 512, h/8, w/8]
        xs_original['1'] = x_rgb
        x_d = self.body_d['layer2'](x_d)
        # x_rgb, x_d, x_fused = self.FusionBlock_1(x_rgb, x_d)
        #xs['1'] = x_fused
        xs['0'] = x_rgb

        x_rgb = self.body['layer3'](x_rgb)   # [2, 1024, h/16, w/16]
        xs_original['2'] = x_rgb
        x_d = self.body_d['layer3'](x_d)
        # x_rgb, x_d, x_fused = self.FusionBlock_2(x_rgb, x_d)
        #xs['2'] = x_fused
        xs['1'] = x_rgb
        
        x_rgb = self.body['layer4'](x_rgb)   # [2, 2048, h/32, w/32]
        xs_original['3'] = x_rgb
        x_d = self.body_d['layer4'](x_d)
        x_rgb, x_d, x_fused = self.FusionBlock_3(x_rgb, x_d)
        #xs['3'] = x_fused
        x_fused = (x_rgb + x_d) / 2
        xs['2'] = x_fused



        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 checkpoint: bool = False,
                 dcn: bool = False):
        norm_layer = FrozenBatchNorm2d
        if checkpoint or dcn:   
            print('Training with checkpoint to save GPU memory.')
            from .resnet import resnet50, resnet101
            if dcn:
                print('Training with dcn.')
                stage_with_dcn = [False, True, True, True]
            else:
                stage_with_dcn = [False, False, False, False]
            backbone = eval(name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer, stage_with_dcn=stage_with_dcn)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)
            
            # backbone for depth
            backbone_d = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer)   #######

        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        # super().__init__(backbone, train_backbone, return_interm_layers)
        super().__init__(backbone, backbone_d, train_backbone, return_interm_layers)  #
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    # return_interm_layers = args.masks or (args.num_feature_levels > 1)
    return_interm_layers = args.num_feature_levels > 1
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.checkpoint, args.dcn)
    model = Joiner(backbone, position_embedding)
    return model
