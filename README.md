# Is Attention All You Need: A Survey about the Efficiency of ViT and Swin Transformer -- CityU GE2340 Project 24/25 semA

## Introduction

We are from CityU GE2340 AI: Past, Present, & Future Group 32. This repository contains some of the code used in our project. We have gathered various libraries that compute the network's MACs and Params, and we found a method for calculating the inference speed of the network.

We would like to thank **THOP**, **fvcore**, **ptflops**, and **calflops** for their contributions. This repository is just a collection.

### Packages Used:
- Python 3.8
- CUDA 11.8
- PyTorch 2.0.0
- timm 1.0.11
- numpy
- tqdm
- Pillow

### Method to Calculate MACs and Params Using Existing Libraries:
```shell
python flops.py
```

### Method to Manually Calculate Model Inference Speed (FPS):
```shell
python cal_time.py
```

Some of our tests on an RTX 3090 are shown in the figure below: ![](result.png)

Datasets and Training
For the evaluators of this project, the PMD_split dataset can be obtained from [PMD](https://jiaying.link/cvpr2020-pgd/). Although they have not released the training code, we implemented the training code based on the descriptions in their paper. We also wrote a dataloader according to the dataset structure. Training can be done with:
```shell
python train_pmd.py
```

Inference can be carried out using the official implementation found at [PMD Official Implementation](https://jiaying.link/cvpr2020-pgd/) if needed.

The dataset for AIGC Detection uses the CS4487 Machine Learning 24/25 semA Project. We do not have the rights to distribute this dataset. Training can be done with:
```shell
python train_AIGC.py
```

Once again, we would like to thank THOP, fvcore, ptflops, calflops, and PMD for their contributions to the open-source community.
