# HKUSTGZ_SegNet

This repo is to conduct semi-supervised semantic segmentation task for Semantic HKUSTGZ Dataset. The followings are usages of different folders:

## datasets

-   Defines class definitions of different datasets (e.g., HKUSTGZ, Cityscapes).
-   Contains utilizations of different pre-processing techniques which are put into the transforms folder.(e.g., data augmentation, uniform sampling).
-   Dataloaders for different datasets are set in the [loader](./datasets.loader). They define the data source, data format to be loaded, as well as the data augmentations for the loaded images.

## models

-   The list of available segmentation models are defined in the loader folder, e.g., [model_manager.py](./models/model_manager.py).

## Losses

-   The list of losses is stored in `SEG_LOSS_DICT` in [loss_manager.py](./loss/loss_manager.py).

## Methods

Available methods: [sdc, ifr]

### Important Paramas of Methods

`contrast`: Use contrastive learning.
`use_proto`: Use prototype networks.

## Evaluators

-   Evaluators are defined in [evaluators](./segmentor/tools/evaluator/__init__.py).

-   Tasks are defined in [task_mapping](./segmentor/tools/evaluator/tasks.py)

## Runtime Acceleration

-   DataParallel / DistributedDataParallel
-   APEX (_TODO_)

## Dataset Preparation

```
$DATA_ROOT
├── hkustgz
│   ├── coarse
│   │   ├── image
│   │   ├── instance
│   │   └── label
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
```
