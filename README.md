# EK-Instance Dataset
This repository includes the codes and data of the EK-Instance dataset, a challenging object instance identification benchmark built on the [EPIC-KITHCENS](https://epic-kitchens.github.io/2022) dataset.  
Please check the paper ["Object Instance Identification in Dynamic Environments"](https://arxiv.org/abs/2206.05319) for details.

<img src="assets/epic22_teaser.png" width="600" >
<img src="assets/epic22_teaser.gif" width="200" >

This repository contains training and evaluation code for object instance identification.

* DataLoader: EK-Instance
* Loss function: Normalized Softmax, ArcFace, N-pair loss
* Clustering: K-means and HAC
* Visualization: Grad-CAM visualization

## Requirements
* Python 3.9+
* gnureadline
* ffmpeg
* numpy
* opencv-python
* pillow
* scikit-learn
* python-Levenshtein
* pycocotools
* [faiss](https://github.com/facebookresearch/faiss)
* torch (1.10.0)
* torchvision (0.11.1)
* [mllogger](https://github.com/takumayagi/mllogger)
* [grad-cam](https://pypi.org/project/grad-cam/)

## Getting Started
### Download the data
1. Download EPIC-KITCHENS-100 videos from the [official site](https://github.com/epic-kitchens/epic-kitchens-download-scripts).
Since this dataset uses 480p frames for training and testing you need to download the original videos.
Place them as data/epic_kitchens/videos/PXX/PXX_XX.MP4.
```
# At the EPIC downloader directory
python epic_downloader.py --videos --participants 1,2,5,7,9,11,14,15,16,18,19,20,25,26,27,29,31,32,33,34,35,36
# At this repository
bash preprocessing/link_video_files.bash <your EPIC-KITCHENS directory>
```

2. Download the annotation from [here](https://drive.google.com/file/d/1D-LjYsHdfjjpDeSkE9pYLRqlDOHXyD7e/view?usp=sharing).
Extract the tar file at data/ (a new directory vott will be created).

```
tar xf reid_anno_211121.tar
```

### Preprocess the data
Extract 480p frames from the video files (this may take time).
```
bash preprocessing/extract_all_frames.bash
```
The above script will create a directory data/epic_kitchens/rgb_frames/.

### Training
Uncomment the necessary lines.
```
bash scripts/train.sh
```

The Training log and trained models will be placed under outputs/\<name\>/.

### Evaluation
Uncomment the necessary lines.  
You may need to change the --resume argument to the model you have trained.
```
bash scripts/test.sh
```

The prediction log and visualization result will be placed under predictions/\<name\>/.

### Evaluation using the pretrained models
You can download the pretrained weights from [this folder](https://drive.google.com/drive/folders/1dT1dG0x3VPCTHrlaWxO275PlIKhfUiSX?usp=sharing).  
You can replicate the results by running the following lines:

| Split: test | HAC threshold | AMI | ACC | Fp | Fb |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ImageNet | 0.25 | 0.892 | 0.799 | 0.789 | 0.828 |
| Normalized Softmax | 0.7 | 0.892 | 0.799 | 0.789 | 0.828 |
| ArcFace | 0.8 | 0.892 | 0.796 | 0.784 | 0.827 |
| Normalized Softmax | 0.6 | 0.924 | 0.854 | 0.847 | 0.874 |

```
# ImageNet baseline without training
python train.py --dataset epic --model ResNet34LateFusion --method paired --batch_size 64 --nb_dims 256 --cluster_alg hac --stride 2 --eval --eval_hac_dists 0.25 --eval_split test
# Normalized softmax
python train.py --dataset epic --model ResNet34MeanProj --method paired --batch_size 64 --nb_dims 256 --cluster_alg hac --stride 2 --eval --resume pretrained/nsoftmax_ResNet34MeanProj_epic.pth --eval_hac_dists 0.7 --eval_split test
# ArcFace
python train.py --dataset epic --model ResNet34MeanProj --method paired --batch_size 64 --nb_dims 256 --cluster_alg hac --stride 2 --eval --resume pretrained/arcface_ResNet34MeanProj_epic.pth --eval_hac_dists 0.8 --eval_split test
# N-pair
python train.py --dataset epic --model ResNet34MeanProj --method paired --batch_size 64 --nb_dims 256 --cluster_alg hac --stride 2 --eval --resume pretrained/paired_ResNet34MeanProj_epic.pth --eval_hac_dists 0.6 --eval_split test
```

## Authors
* Takuma Yagi (tyagi[at]iis.u-tokyo.ac.jp)
* Md Tasnimul Hasan (tasnim[at]iis.u-tokyo.ac.jp)
* Yoichi Sato (ysato[at]iis.u-tokyo.ac.jp)

## Citation
```
@inproceedings{yagi2022object,
  title = {Object Instance Identification in Dynamic Environments},
  author = {Yagi, Takuma and Hasan, Md Tasnimul and Sato, Yoichi},
  booktitle = {Proceedings of the Tenth International Workshop on Egocentric Perception, Interaction and Computing (EPIC 2022 at CVPR 2022, extended abstract)},
  doi = {},
  pages = {},
  url = {},
  year = {2022}
} 
```

When you use the data for training and evaluation, please also cite the original dataset ([EPIC-KITCHENS Dataset](https://epic-kitchens.github.io/)).
