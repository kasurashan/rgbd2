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
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import functools
print = functools.partial(print, flush=True)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if utils.is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model,data_loader, device):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import cv2

    model.eval()
    

    cnt = 0
    for samples, targets in data_loader:
        cnt += 1
        samples = samples.to(device)
        # print(targets)

        conv_features = []
        hook = model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output))

        outputs = model(samples)    
        # print(model.backbone)
        hook.remove()
        
        conv_features = conv_features[0]  # 0, 1, 2 dict
        # print(conv_features['0'].tensors.shape)  # 1, 512, 100, 134
        # print(model.backbone[-2])
        # get the feature map shape
        
        h, w = conv_features['0'].tensors.shape[-2:]

        x0 = conv_features['0'].tensors.squeeze(0)
        x0 = np.array(torch.mean(x0, dim=0).cpu())
        # print(x.shape)
        x1 = conv_features['1'].tensors.squeeze(0)
        x1 = np.array(torch.mean(x1, dim=0).cpu())

        x2 = conv_features['2'].tensors.squeeze(0)
        x2 = np.array(torch.mean(x2, dim=0).cpu())

        

        # print(np.array(samples.tensors.squeeze().cpu())) # 3 h w

        samples.tensors = samples.tensors.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        # 정규화를 되돌림
        # print(samples.tensors.shape)
        samples.tensors = samples.tensors[:,:3,:,:] * std.unsqueeze(1).unsqueeze(1) + mean.unsqueeze(1).unsqueeze(1)


        img = np.array(samples.tensors.squeeze())
        # print(img.shape, img)
        img = np.transpose(img, (1, 2, 0))       



        # 서브플롯 생성
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        # 첫 번째 이미지 표시
        ax1.imshow(img)
        ax1.axis('off')  # 축 숨기기

        # 두 번째 이미지 표시
        ax2.imshow(x0)
        ax2.axis('off')  # 축 숨기기


        ax3.imshow(x1)
        ax3.axis('off')  # 축 숨기기


        ax4.imshow(x2)
        ax4.axis('off')  # 축 숨기기

        # print(model.backbone[-2]) # bacbonbe 융합되고 최종
        # print(model.backbone[-2].body) # rgb backbone
        # print(model.backbone[-2].body_d) # depth backbone
        
        # # 이미지 저장
        plt.savefig(f'./img_vis_sunrgbd_rgb/subplots_image{cnt}.png', dpi=500)
        # break

    return