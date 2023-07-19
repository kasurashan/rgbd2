EXP_DIR=exps/solq_rgb.r50
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
       --use_env /root/workspace/SOLQ/main.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path  ../../datasets/nyuv2/ \
       --batch_size 1 \
       --vector_hidden_dim 1024 \
       --resume /root/workspace/SOLQ/solq_r50_final.pth \
       --vector_loss_coef 3 \
       --epochs 50 \
       --lr_drop 15 \
       --output_dir ${EXP_DIR}

EXP_DIR=exps/solq.r50
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
       --use_env /root/workspace/SOLQ_rgbd/main.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path  ../../datasets/nyuv2/ \
       --batch_size 1 \
       --vector_hidden_dim 1024 \
       --resume /root/workspace/SOLQ_rgbd/solq_r50_final.pth \
       --vector_loss_coef 3 \
       --epochs 50 \
       --lr_drop 15 \
       --depth_data nyu \
       --copy_backbone True \
       --output_dir ${EXP_DIR}


