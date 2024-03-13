EXP_DIR=exps/solq2.r50
CUDA_VISIBLE_DEVICES=9 python main_vis.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path /root/datasets/sun_mini \
       --batch_size 1 \
       --vector_hidden_dim 1024 \
       --vector_loss_coef 3 \
       --output_dir ${EXP_DIR} \
       --depth_data nyu \
       --resume /root/workspace/solq_rgbd/exps/solq.r50_sunrgbd_rgbd_ours_fusion3e-5/checkpoint_best_segm.pth \
       --eval