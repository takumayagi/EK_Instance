#! /bin/sh
#
# train.sh
# Copyright (C) 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

#BATCH_SIZE=448  # For A100 node (40GB)
BATCH_SIZE=96  # For typical GPUs (16GB)

# Normalized Softmax
python train.py --method softmax --dataset epic --batch_size $BATCH_SIZE --track_per_class 4 --iter_display 125 --iter_snapshot 125 --iter_evaluation 125 --nb_iters 1500 --lr_step_list 1000 1250 --save_model --model ResNet34MeanProjNormalizedSoftmax --dir_name baseline_256 --cpu --nb_dims 256 --cluster_alg hac --lr 3e-4 --tau 0.2 --eval_hac_dists 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625

# ArcFace
#python train.py --method arcface --dataset epic --batch_size $BATCH_SIZE --track_per_class 4 --iter_display 125 --iter_snapshot 250 --iter_evaluation 250 --nb_iters 7500 --lr_step_list 4500 6000 --save_model --model ResNet34MeanProj --dir_name baseline_256 --cpu --nb_dims 256 --cluster_alg hac --lr 3e-5 --eval_hac_dists 0.55 0.6 0.65 0.7 0.75 0.8 0.85

# N-paired
#python train.py --method paired --dataset epic --batch_size $BATCH_SIZE --track_per_class 4 --iter_display 125 --iter_snapshot 2250 --iter_evaluation 2250 --nb_iters 2500 --lr_step_list 1500 2250  --save_model --model ResNet34MeanProj --dir_name baseline_256 --cpu --nb_dims 256 --stride 2 --track_per_class 4 --tau 0.07 --cluster_alg hac --lr 3e-5 --eval_hac_dists 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7
