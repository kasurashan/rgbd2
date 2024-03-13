# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
import functools
print = functools.partial(print, flush=True)
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import numpy as np   # depth npy load
from PIL import Image ##


class CocoDetection(TvCocoDetection):
    # def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
    #     super(CocoDetection, self).__init__(img_folder, ann_file,
    #                                         cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
    #     self._transforms = transforms
    #     self.prepare = ConvertCocoPolysToMask(return_masks)

    # def __getitem__(self, idx):
    #     img, target = super(CocoDetection, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     target = {'image_id': image_id, 'annotations': target}
    #     img, target = self.prepare(img, target)
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #     return img, target

    def __init__(self, img_folder, ann_file, depth_folder, transforms, return_masks, depth_data, cache_mode=False, local_rank=0, local_size=1): ##
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.depth_folder = depth_folder  
        self.depth_data = depth_data  

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']

        if self.depth_data == 'box':
            depth_file = img_file_name.rstrip('.png').rstrip('.jpg') + '.npy'
            depth_data_path = str(self.depth_folder) + '/' + depth_file
            depth_data = np.load(depth_data_path)
            try:
                depth_data = depth_data.squeeze()
            except:
                pass
            depth_img = Image.fromarray(depth_data, mode='F')
            
        if self.depth_data == 'nyu':
            depth_file = img_file_name.lstrip('train/color/nyu_rgb_').lstrip('val/color/nyu_rgb_')
            depth_file = depth_file.zfill(10)  # 10자리로 맞춰주면서 왼쪽에 0 추가
            depth_data_path = str(self.depth_folder) + '/' + depth_file
            depth_data = Image.open(depth_data_path)
            depth_img = np.array(depth_data).astype(np.float32)
            depth_img = depth_img / np.max(depth_img) * 255
            depth_img = Image.fromarray(depth_img)

        if self.depth_data == 'scan':
            depth_file = img_file_name[0:12] + '/depth/' + img_file_name[-10:-4] + '.png'
            depth_data_path = str(self.depth_folder) + '/' + depth_file
            depth_data = Image.open(depth_data_path)
            depth_img = np.array(depth_data).astype(np.float32)
            depth_img = depth_img / np.max(depth_img) * 255
            depth_img = Image.fromarray(depth_img)
        
        
        
        
        target = {'image_id': image_id, 'annotations': target}

        
        img, target = self.prepare(img, target)

        img = [img, depth_img]  ##
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        img = torch.tensor(np.concatenate((np.array(img[0]), np.array(img[1])), axis=0))   # [4, h, w]  concat해서 채널 4됨
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # for Swin-L
    # scales = [416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    # for Swin-L
    # if image_set in ['val', 'test']:
    #     return T.Compose([
    #         T.RandomResize([1088], max_size=1333),
    #         normalize,
    #     ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    #     'test': (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json'),
    # }
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json', root / "train2017_depth"),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json', root / "val2017_depth"),
    }

    if args.depth_data == 'scan':
        PATHS = {
            "train": (root / "SCANNET_SEMSEG", root / "annotations" / f'{mode}_train2017.json', root / "SCANNET_SEMSEG"),
            "val": (root / "SCANNET_SEMSEG", root / "annotations" / f'{mode}_val2017.json', root / "SCANNET_SEMSEG"),
        }


    if args.eval and args.test:
        print('Inference on test-dev.')
        image_set = 'test'
    # img_folder, ann_file = PATHS[image_set]
    img_folder, ann_file, depth_folder = PATHS[image_set] ##
    # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
    #                         cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    dataset = CocoDetection(img_folder, ann_file, depth_folder, transforms=make_coco_transforms(image_set), return_masks=args.masks, depth_data=args.depth_data,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    return dataset




def build_original(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set] ##
    dataset = CocoDetection_original(img_folder, ann_file, transforms=None, return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    
    return dataset


class CocoDetection_original(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_original, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection_original, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target