EXP_DIR=exps/solq.r50_lossvis
CUDA_VISIBLE_DEVICES=3 python /root/workspace/backup/workspace/SOLQ_rgbd4/main_landscape.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path  /root/datasets/nyuv2 \
       --batch_size 8 \
       --vector_hidden_dim 1024 \
       --resume /root/workspace/backup/workspace/SOLQ_rgbd/exps/solq.r50nat_nyu/checkpoint_best_segm.pth \
       --resume2 /root/workspace/backup/workspace/SOLQ_rgbd4/exps/solq.r50_onlyrgb/checkpoint_best_segm.pth \
       --resume3 /root/workspace/backup/workspace/SOLQ_rgbd4/exps/solq.r50_onlydepth/checkpoint_best_segm.pth \
       --vector_loss_coef 3 \
       --depth_data nyu \
       --copy_backbone False \
       --eval \
       --output_dir ${EXP_DIR}
