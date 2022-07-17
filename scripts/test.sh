#! /bin/sh
#
# test.sh
# Copyright (C) 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.
#

#BATCH_SIZE=112
BATCH_SIZE=64  # Works in most GPUs

# Sample code on "ImageNet" model

# In using K-means, you don't need perform hyperparameter tuning:
#python train.py --method paired --batch_size $BATCH_SIZE --cpu --nb_dims 512 --cluster_alg kmeans --eval --model ResNet34LateFusion --dir_name imagenet_km

# In using HAC, you have to first determine the best threshold in validation set
#python train.py --method paired --batch_size $BATCH_SIZE --cpu --nb_dims 512 --cluster_alg hac --eval --model ResNet34LateFusion --eval_hac_dists 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 --test_pids configs/valid_pids.txt

# When testing you should use single threshold to determine the final performance
#python train.py --method paired --batch_size $BATCH_SIZE --cpu --nb_dims 512 --cluster_alg hac --eval --model ResNet34LateFusion --eval_hac_dists 0.25 --test_pids configs/test_pids.txt --dir_name imagenet

# Visualization
#python train.py --method paired --batch_size $BATCH_SIZE --cpu --nb_dims 512 --cluster_alg hac --eval --model ResNet34LateFusion --eval_hac_dists 0.25 --test_pids configs/test_pids.txt --dir_name imagenet --vis

# Specify --resume if you want to load trained model
python train.py --dataset epic --batch_size $BATCH_SIZE --cpu --cluster_alg hac --eval --eval_hac_dists 0.6 --resume outputs/softmax_ResNet34MeanProjNormalizedSoftmax_epic_baseline_256_211201/model_000500.pth --dir_name proposed
