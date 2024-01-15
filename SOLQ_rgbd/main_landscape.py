# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine_landscape import evaluate
from models import build_model
from util.tool import load_model
from util.dist import configure_nccl, init_process_group, synchronize
from util.log import setup_logger, setup_writer
import functools
import copy

print = functools.partial(print, flush=True)

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_period', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--meta_arch', default='solq', type=str)


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # VecInst
    parser.add_argument('--with_vector', default=False, action='store_true')
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--vector_loss_coef', default=0.7, type=float)
    parser.add_argument('--vector_hidden_dim', default=256, type=int,
                        help="Size of the vector embeddings (dimension of the transformer)")
    parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--vector_start_stage', default=0, type=int)
    parser.add_argument('--num_machines', default=1, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--dcn', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--depth_data', default='nyu')   ######################## add for nyu, box  scan data (depth dataload process is little different)

    parser.add_argument('--alg', default='instformer', type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume2', default='', help='resume from checkpoint') ####################################
    parser.add_argument('--resume3', default='', help='resume from checkpoint') ####################################


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--copy_backbone', default=False)   ############### 처음에 학습시에 RGB branch의 weight를  depth branch 에 복사하기 위해 

    # distributed
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist-url', default=None, type=str, help='url used to set up distributed training')
    parser.add_argument('--rank', default=None, type=int, help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--num-machines', default=None, type=int)
    return parser


def main(gpu, ngpus_per_node, args):
    if args.num_machines == 1:
        utils.init_distributed_mode(args)
    else:
        configure_nccl()
        # ------------ set environment variables for distributed training ------------------------------------- #
        if args.rank is None:
            args.rank = int(os.getenv('RLAUNCH_REPLICA', '0'))

        os.environ['LOCAL_RANK'] = '0'
        os.environ['LOCAL_SIZE'] = '1'
        args.distributed = True

        args.gpu = gpu
        if ngpus_per_node > 1:
            args.rank = args.rank * ngpus_per_node + gpu

            # initialize process group
            init_process_group(args)
        synchronize()

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    if args.num_machines == 1:
        device = torch.device(args.device)
    else:
        device = torch.device(args.gpu)
        args.device = args.gpu # fix multi-machines launch problems

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    if args.num_machines == 1:
        model.to(device)
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        print('Training with distributed.')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)  #############
        synchronize()
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # if args.pretrained is not None:
    #     model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')   ###########
            checkpoint2 = torch.load(args.resume2, map_location='cpu') ################### 
            checkpoint3 = torch.load(args.resume3, map_location='cpu') ################# 

            

        print(checkpoint.keys())
        print(checkpoint['model'].keys())

        for k in checkpoint['model'].keys():
            # print(checkpoint['model'][k])
            # print(checkpoint2['model'][k])
            # checkpoint['model'][k] = (checkpoint['model'][k] + checkpoint2['model'][k] + checkpoint3['model'][k]) / 3
            # checkpoint['model'][k] = (checkpoint['model'][k] + checkpoint3['model'][k]) / 2
            pass
            # print(checkpoint['model'][k])

        
        check_keys = checkpoint['model'].keys() # ordered dict



        def group_product(xs, ys):
            """
            the inner product of two lists of variables xs,ys
            :param xs:
            :param ys:
            :return:
            """
            return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])



        def normalization(v):
            """
            normalization of a list of vectors
            return: normalized vectors v
            """
            s = group_product(v, v)
            s = s**0.5
            s = s.cpu().item()
            v = [vi / (s + 1e-6) for vi in v]
            return v


        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))



        o = [] # 그냥 기준이 되는 rgbd weight
        u = [] # rgb - rgbd 방ㅇ향
        v = [] # depth - rgbd 방향
        u_interpolate = [] # rgb rgbd 평균
        v_interpolate =  [] # depth rgbd 평균
        uv_interpolate = [] # rgb depth 평균
        tot_interpolate = [] # rgbd rgb depth 평균
        for k in checkpoint['model'].keys():
            o.append(checkpoint['model'][k])
            u.append(checkpoint2['model'][k] - checkpoint['model'][k])
            v.append(checkpoint3['model'][k] - checkpoint['model'][k])

            u_interpolate.append((checkpoint2['model'][k] - checkpoint['model'][k])/2)
            v_interpolate.append((checkpoint3['model'][k] - checkpoint['model'][k])/2)
            uv_interpolate.append((checkpoint2['model'][k] + checkpoint3['model'][k])/2 - checkpoint['model'][k])
            tot_interpolate.append((checkpoint['model'][k] + checkpoint2['model'][k] + checkpoint3['model'][k])/3 - checkpoint['model'][k])
            # print(checkpoint['model'][k])
            # print(checkpoint2['model'][k])
            # checkpoint['model'][k] = (checkpoint['model'][k] + checkpoint2['model'][k] + checkpoint3['model'][k]) / 3
            # checkpoint['model'][k] = (checkpoint['model'][k] + checkpoint3['model'][k]) / 2
            # print(checkpoint['model'][k])
        
        u_ori = copy.deepcopy(u)
        v_ori = copy.deepcopy(v)

        dx = np.sqrt(sum([np.dot(u1.cpu().numpy().ravel(), u2.cpu().numpy().ravel()) for u1, u2 in zip(u, u)]))

        u /= dx
        
        # u = normalization(u) # [tensor1, tensor2, ....] 서로 다른 크기의 텐서(필터)들의 리스트

        ss = [np.dot(uu.cpu().numpy().ravel(), vv.cpu().numpy().ravel()) for uu, vv in zip(u, v)]
        ss = sum(ss)
        v = [x-ss*y for x,y in zip(v,u)]
        dy = np.sqrt(sum([np.dot(v1.cpu().numpy().ravel(), v2.cpu().numpy().ravel()) for v1, v2 in zip(v, v)]))

        v /= dy


        # v = normalization(v)  # [tensor1, tensor2, ....] 서로 다른 크기의 텐서(필터)들의 리스트


        # 2차원 loss landscaep에서 각 weight들이 어디에 위치할지 u,v에 정사영시켜서 x,y좌표 얻어내기
        # 어차피 rgbd의 weight o는 (0,0)이니까 고려 안 해도 됨

        def get_xy(u_ori, u, v):
            u_x = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(u_ori, u)])
            u_y = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(u_ori, v)])
            return np.array([u_x, u_y])

        bend_coordinates = np.stack((get_xy(u_ori, u, v), get_xy(v_ori, u, v), get_xy(u_interpolate, u, v), get_xy(v_interpolate, u, v), get_xy(uv_interpolate, u, v), get_xy(tot_interpolate, u, v)))



        # u_x = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(u_ori, u)])
        # u_y = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(u_ori, v)])

        # v_x = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(v_ori, u)])
        # v_y = sum([np.dot(x.cpu().numpy().ravel(), y.cpu().numpy().ravel()) for x, y in zip(v_ori, v)])

        curve_points = 61
        ts = np.linspace(0.0, 1.0, curve_points)
        # curve_coordinates = []
        # for t in np.linspace(0.0, 1.0, args.curve_points):
        #     weights = curve_model.weights(torch.Tensor([t]).cuda())
        #     curve_coordinates.append(get_xy(weights, w[0], u, v))
        # curve_coordinates = np.stack(curve_coordinates)

        grid_points = 5 # default = 21인데 1번도는데 한 1분걸리는듯
        margin_left = 0.2
        margin_right = 0.2
        margin_bottom = 0.2
        margin_top = 0.2 
        G = grid_points
        alphas = np.linspace(0.0 - margin_left, 1.0 + margin_right, G)
        betas = np.linspace(0.0 - margin_bottom, 1.0 + margin_top, G)
        
        te_loss = np.zeros((G, G))
        grid = np.zeros((G, G, 2))
    



             

        
        

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                p_perturb = [x + alpha * y * dx + beta * z * dy for x,y,z in zip(o,u,v)]

                p_perturb = dict((k, v) for k, v in zip(check_keys, p_perturb))

                for name, parameter in model.named_parameters():

                    # parameter.data = p
                    # print(parameter.data.shape==p_perturb[name].shape)  # 이게,.,. 순서가 섞여있어서 key로 조회해서 대응시켜야해..
                    parameter.data = p_perturb[name].cuda()
                    

                    

                
                

                te_loss_v = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
                # te_loss_v = evaluate(model, criterion, postprocessors, data_loader_train, base_ds, device, args.output_dir)



                grid[i, j] = [alpha * dx, beta * dy]


                te_loss[i, j] = te_loss_v

        
        dir = '/root/workspace/backup/workspace/SOLQ_rgbd4/'
        np.savez(
            os.path.join(dir, 'plane.npz'),
            ts=ts,
            bend_coordinates=bend_coordinates,
            alphas=alphas,
            betas=betas,
            grid=grid,
            te_loss=te_loss
            )




        

    



if __name__ == '__main__':
    import os
    os.environ["NCCL_P2P_DISABLE"] = "1"
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # multi-processing
    if args.num_machines is None:
        args.num_machines = int(os.getenv('RLAUNCH_REPLICA_TOTAL', '1'))

    print('Total number of using machines: {}'.format(args.num_machines))
    ngpus_per_node = torch.cuda.device_count()
    if args.num_machines > 1 and ngpus_per_node > 1:
        args.world_size = ngpus_per_node * args.num_machines
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        main(0, ngpus_per_node, args)
        

