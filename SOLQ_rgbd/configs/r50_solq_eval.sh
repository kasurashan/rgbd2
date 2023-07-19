

EXP_DIR=exps/solq.r50
python -m torch.distributed.launch --nproc_per_node=2 \
       --use_env main.py \
       --meta_arch solq \
       --with_vector \
       --with_box_refine \
       --masks \
       --coco_path ../../datasets/nyuv2/ \
       --batch_size 1 \
       --vector_hidden_dim 1024 \
       --vector_loss_coef 3 \
       --output_dir ${EXP_DIR} \
       --resume ${EXP_DIR}/checkpoint.pth \
       --depth_data nyu \
       --eval