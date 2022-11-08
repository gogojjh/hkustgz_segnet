# HKUSTGZ_SegNet

This repo is to conduct semi-supervised semantic segmentation task for Semantic HKUSTGZ Dataset.

## Dataset

-   Defines class definitions of different datasets (e.g., HKUSTGZ, Cityscapes).
-   Contains utilizations of different pre-processing techniques which are put into the transforms folder.(e.g., data augmentation, uniform sampling).

## Network

-   Contains different semantic segmentation backbones for testing.

## Loss

-   List of losses are stored in SEG_LOSS_DICT in [loss_manager.py](loss/loss_manager.py)

## Runtime Acceleration

-   DDP
-   APEX (_TODO_)
