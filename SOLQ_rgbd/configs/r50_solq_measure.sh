EXP_DIR=exps/solq.r50_nyu_measure
CUDA_VISIBLE_DEVICES=8 python main_measure.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path  /root/datasets/nyu \
       --batch_size 1 \
       --vector_hidden_dim 1024 \
       --depth_data nyu \
       --vector_loss_coef 3 \
       --epochs 1 \
       --lr_drop 15 \
       --copy_backbone True \
       --throughput \
       --fps \
       --output_dir ${EXP_DIR}
