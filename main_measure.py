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
from engine_measure import evaluate, train_one_epoch ###################################
from models import build_model
from util.tool import load_model
from util.dist import configure_nccl, init_process_group, synchronize
from util.log import setup_logger, setup_writer

import functools

print = functools.partial(print, flush=True)

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_backbone_depth', default=2e-5, type=float)    # depth backbone 파트에 대한 러닝레이트를 따로
    parser.add_argument('--lr_backbone_fusion', default=2e-5, type=float)   # Fusion 파트에 대한 러닝레이트를 따로
    
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


     # measure
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--fps', action='store_true')

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


    # original param dicts
    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr * args.lr_linear_proj_mult,
    #     }
    # ]


    # fusion module의 lr도 따로 조절하기 위해
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad and not match_name_keywords(n, ["body_d"]) and not match_name_keywords(n, ["Fusion"])],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, ["body_d"]) and p.requires_grad],
            "lr": args.lr_backbone_depth,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, ["Fusion"]) and p.requires_grad],
            "lr": args.lr_backbone_fusion,
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
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        # checkpoint에서 진짜로 이어서 학습하고 싶은 경우 (50epoch학습시키려고 했는데 중간에 40까지밖에 못해서 이어서 하기 위해)
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     import copy
        #     p_groups = copy.deepcopy(optimizer.param_groups)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for pg, pg_old in zip(optimizer.param_groups, p_groups):
        #         pg['lr'] = pg_old['lr']
        #         pg['initial_lr'] = pg_old['initial_lr']
        #     #print(optimizer.param_groups)
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
        #     args.override_resumed_lr_drop = True
        #     if args.override_resumed_lr_drop:
        #         print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
        #         lr_scheduler.step_size = args.lr_drop
        #         lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        #     lr_scheduler.step(lr_scheduler.last_epoch)
        #     args.start_epoch = checkpoint['epoch'] + 1
        #####################################################################################################



        # check the resumed model
        if not args.eval:
            
            # check the resumed model 
            # test_stats, coco_evaluator = evaluate(
            #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            # )
            
            if args.copy_backbone:   #### 처음부터 학습할때  비어있는 depth쪽에 copy해서 넣어줌
                import copy
                #print(checkpoint['model'])  # ordered dict
                weight_copy = copy.deepcopy(checkpoint['model'])
                #for k, v in checkpoint['model'].items():
                for k,v in weight_copy.items():
                    #print(k) # 이름 detr.backbone.0.body.layer3.5.bn3.running_var
                    #print(v) # 실제 텐서
                    if k[0:8] == 'backbone':
                        
                        new_key = 'backbone.0.body_d' + k[15:]
                        print(new_key)
                        new_value = v
                        checkpoint['model'].update({new_key : new_value})

    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            # here dump results in json format
            image_set = 'test-dev' if args.test else 'val' 
            print('Dump results to {}.'.format(os.path.join(args.output_dir, f'detections_{image_set}2017_{args.alg}_results.json')))
            with open(os.path.join(args.output_dir, f'detections_{image_set}2017_{args.alg}_results.json'), 'w', encoding='utf-8') as f:
                json.dump(coco_evaluator.eval_results, f, cls=utils.AdaptiveEncoder, indent=4)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm , args.throughput) #########
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.fps:
        fps = compute_fps(model, dataset_val, num_iters=300, batch_size=1)
        print(fps)


from util.misc import nested_tensor_from_tensor_list
from tqdm import tqdm
@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    # note that warm-up iters. are excluded from the total iters.
    for iter_ in tqdm(range(warm_iters + num_iters)):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    return sum(ts) / len(ts)

    
@torch.no_grad()
def compute_fps(model, dataset, num_iters=300, warm_iters=5, batch_size=4):
    print(f"computing fps.. (num_iters={num_iters}, batch_size={batch_size}) "
          f"warm_iters={warm_iters}, batch_size={batch_size}]")
    assert num_iters > 0 and warm_iters >= 0 and batch_size > 0
    model.cuda()
    model.eval()
    inputs = nested_tensor_from_tensor_list(
        [dataset.__getitem__(0)[0].cuda() for _ in range(batch_size)])
    t = measure_average_inference_time(model, inputs, num_iters, warm_iters)
    model.train()
    print(f"FPS: {1.0 / t * batch_size}")  
    return 1.0 / t * batch_size




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
        

