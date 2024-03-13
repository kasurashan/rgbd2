# EXP_DIR=exps/solq2.r50
# CUDA_VISIBLE_DEVICES=0 python main_hook.py \
#        --meta_arch solq \
#        --with_vector \
#        --with_box_refine \
#        --masks \
#        --coco_path /root/datasets/box \
#        --batch_size 1 \
#        --vector_hidden_dim 1024 \
#        --vector_loss_coef 3 \
#        --output_dir ${EXP_DIR} \
#        --depth_data box \
#        --resume /root/workspace/solq_rgbd/exps/solq.r50_box_ours/checkpoint_best_segm.pth \
#        --eval

# EXP_DIR=exps/solq2.r50
# CUDA_VISIBLE_DEVICES=0 python main_hook.py \
#        --meta_arch solq \
#        --with_vector \
#        --with_box_refine \
#        --masks \
#        --coco_path /root/datasets/sunrgbd \
#        --batch_size 1 \
#        --vector_hidden_dim 1024 \
#        --vector_loss_coef 3 \
#        --output_dir ${EXP_DIR} \
#        --depth_data nyu \
#        --resume /root/workspace/solq_rgbd/exps/solq.r50_sunrgbd_rgbd_ours_fusion3e-5/checkpoint_best_segm.pth \
#        --eval