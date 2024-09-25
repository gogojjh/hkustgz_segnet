# HKUSTGZ_SegNet_ROS

Semantic Segmentation for Semantic HKUSTGZ Dataset.

## Installation

### Docker Image

```
docker login -u iidcramlab -p ramlab-123
docker pull iidcramlab/segnet
```

### Dockerfile (Pass SSH Key to git pull private repo)

```
docker build -f dockerfile_ros . -t hkustgz_segnet_ros --build-arg SSH_PRIVATE_KEY="-----BEGIN OPENSSH PRIVATE KEY-----
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
-----END OPENSSH PRIVATE KEY-----"
```

### Docker Container

```
docker run -it --gpus all --shm-size 204g -v /data/rygeng:/data -v /home/rygeng:/save_data --name hkustgz_segnet_ros janeandrebecca/hkustgz_segnet_ros /bin/zsh
```

## Train/Validation/Test without ROS

```
cd /home/hkustgz_segnet/scripts/cityscapes/hrnet
# train
sh run_h_48_d_4_prob_proto.sh train "fast_mls" "/data" "/save_data"
# test
sh run_h_48_d_4_prob_proto.sh test "fast_mls" "/data" "/save_data"
```

Please refer to [openseg](https://github.com/openseg-group/openseg.pytorch) for more details about the usage of the scripts.

## Inference with ROS

```
roscore &
cd ros_workspace
catkin build segnet_ros
source devel/setup.zsh
cd ros_workspace/src/HKUSTGZ_SegNet/segnet_ros
roslaunch segnet_ros segnet_ros.launch
```

### Configure the Running Scripe

In [run_segnet_ros.sh](segnet_ros/scripts/run_segnet_ros.sh):

```
# change the checkpoint path
CHECKPOINTS_ROOT="/data/checkpoints"
CHECKPOINTS_NAME="hr_w48_attn_uncer_proto_hkustgz_max_performance.pth"
```

### Config for ROS

In [hkustgz_ros.json](segnet/configs/hkustgz/hkustgz_ros.json):

```
# change number of classes according to corresponding yaml/json file
"data": {
		"num_classes": 23
		}

# change model path
"network": {
		"resume": "/data/checkpoints/hr_w48_attn_uncer_proto_hkustgz_max_performance.pth",
	}

# change ros topics
"ros": {
		"use_ros": true,
		"image_topic": "/camera1/image_color/compressed",
		"sem_image_topic": "/semantic_image",
		"uncer_image_topic": "/uncertainty_image",
		"msg_type": "sensor_msgs/CompressedImage"
	}

# save result path
"test": {
		"out_dir": "/data/hkustgz_result",
		"vis_pred": true
	}
```

## Config Versions
### Current Config
version_3([fusionportable_cityscapes_v3.json](segnet/configs/hkustgz/v3/fusionportable_cityscapes_v3.json)/[fusionportable_cityscapes_v3.yaml](segnet/configs/hkustgz/v3/fusionportable_cityscapes_v3.yaml))

### version_1([semantic_fusionportable_v1.json](segnet/configs/hkustgz/v1/semantic_fusionportable_v1.json)/[semantic_fusionportable_v1.yaml](segnet/configs/hkustgz/v1/semantic_fusionportable_v1.yaml))

Training with 25 classes of Fusionportable in total, and the training data only includes Fusionportable.

### version_2([fusionportable_cityscapes_v2.json](segnet/configs/hkustgz/v2/fusionportable_cityscapes_v2.json)/[fusionportable_cityscapes_v2.yaml](segnet/configs/hkustgz/v2/fusionportable_cityscapes_v2.yaml))

Integrate some classes of the original Fusionportable dataset into one class for training, and the resulting number of classes is 23. Training data includes both Fusionportable and Cityscapes (class_id = 15: 'bus', class_id = 16: 'train' only exist in Cityscapes).

-   'parking' and 'low-speed road' are integrated into 'drivable road'
-   'bike path' is integrated into 'sidewalk'
-   'road marking' is integrated into 'lane'

### version_3([fusionportable_cityscapes_v3.json](segnet/configs/hkustgz/v3/fusionportable_cityscapes_v3.json)/[fusionportable_cityscapes_v3.yaml](segnet/configs/hkustgz/v3/fusionportable_cityscapes_v3.yaml))

Integrate some classes of the original Fusionportable dataset into one class for training, and the resulting number of classes is 23. Training data includes both Fusionportable and Cityscapes (class_id = 15: 'bus', class_id = 16: 'train' only exist in Cityscapes).

-   'parking', 'low-speed road', 'road marking', 'lane' are integrated into 'drivable road'

## Usages of different folders:

### Datasets

-   Defines class definitions of different datasets (e.g., HKUSTGZ, Cityscapes).
-   Contains utilizations of different pre-processing techniques which are put into the transforms folder.(e.g., data augmentation, uniform sampling).
-   Dataloaders for different datasets are set in the [loader](./datasets.loader). They define the data source, data format to be loaded, as well as the data augmentations for the loaded images.

### Models

-   The list of available segmentation models are defined in the loader folder, e.g., [model_manager.py](./models/model_manager.py).

### Losses

-   The list of losses is stored in `SEG_LOSS_DICT` in [loss_manager.py](./loss/loss_manager.py).

## Dataset Preparation

```
$DATA_ROOT
├── hkustgz
│   ├── train
│   │   ├── image
│   │   └── label_id
│   ├── val
│   │   ├── image
│   │   └── label_id
```

## Current Results

Training data: Cityscapes
Test data: HKUSTGZ
![image](docs/media/rviz_screenshot.png)
