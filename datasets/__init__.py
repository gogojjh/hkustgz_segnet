""" 
Dataset setup and loaders
Construct different dataset with different data augmentation methods according to user params.
"""

from datasets import hkustgz, cityscapes

import torch
import torchvision.transforms as standard_transforms
import transforms.joint_transforms as joint_transforms
import transforms.transforms as extended_transforms
from torch.utils.data import DataLoader


def setup_transform(configs, args):
    """ 
    Set up transform lists for training and validation.
    joint_transform: Joint transform of both imgs and labels in 'Improving Semantic Segmentation via Video Propagation and Label Relaxation'.
    """
    if configs['transform']['joint_transform']:
        # Geometric image transformations
        train_joint_transform_list = [
            joint_transforms.RandomSizeAndCrop(configs['transform']['joint_transform']['crop_size'],
                                               False,
                                               pre_size=configs['transform']['joint_transform']['pre_size'],
                                               scale_min=configs['transform']['joint_transform']['scale_min'],
                                               scale_max=configs['transform']['joint_transform']['scale_max'],
                                               ignore_index=args.dataset_cls.ignore_label),
            joint_transforms.Resize(
                configs['transform']['joint_transform']['crop_size']),
            joint_transforms.RandomHorizontallyFlip()]
        train_joint_transform = joint_transforms.Compose(
            train_joint_transform_list)

        # Image appearance transformations
        train_input_transform = []
        if configs['transform']['joint_transform']['color_aug']:
            train_input_transform += [extended_transforms.ColorJitter(
                brightness=configs['transform']['joint_transform']['color_aug'],
                contrast=configs['transform']['joint_transform']['color_aug'],
                saturation=configs['transform']['joint_transform']['color_aug'],
                hue=configs['transform']['joint_transform']['color_aug']
            )]

        if configs['transform']['joint_transform']['bilateral_blur']:
            train_input_transform += [extended_transforms.RandomBilateralBlur()]
        elif configs['transform']['joint_transform']['gaussian_blur']:
            train_input_transform += [extended_transforms.RandomGaussianBlur()]
        else:
            pass

        # Normalize transformation
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_input_transform += [standard_transforms.ToTensor(),
                                  standard_transforms.Normalize(*mean_std)]
        train_input_transform = standard_transforms.Compose(
            train_input_transform)

        val_input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        target_transform = extended_transforms.MaskToTensor()

        # transform of the mask
        # TODO: boundary label relaxation
        target_train_transform = extended_transforms.MaskToTensor()

        return train_joint_transform, train_input_transform, target_train_transform, val_input_transform, target_transform


def setup_loaders(configs, args):
    """ 
    Set up data loader for HKUSTGZ dataset.
    """

    if configs['dataset']['name'] in ['hkustgz', 'hkustgz', 'mapillary']:
        args.dataset_cls = configs['dataset']['name']
    else:
        raise Exception('Dataset {} is not supported.'.format(
            configs['dataset']['name']))

    # Set up training data transform.
    train_joint_transform, train_input_transform, target_train_transform, val_input_transform, target_transform = setup_transform(
        configs, args)
    # TODO: Uniform sampling of the dataset
    if configs['dataset']['name'] == 'hkustgz':
        data_quality = 'fine'
        mode = 'train'

        train_set = args.dataset_cls.HKUSTGZ(
            quality=data_quality,
            mode=mode,
            maxSkip=configs['dataset']['max_skip'],
            joint_transorm=train_joint_transform,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=configs['dataset']['dump_aug_img'],
            cv_split=configs['dataset']['cv_split'])

        val_set = args.dataset_cls.HKUSTGZ(quality='fine',
                                           mode='val',
                                           maxSkip=0,
                                           transform=val_input_transform,
                                           target_transform=target_train_transform,
                                           cv_split=configs['dataset']['cv_split']
                                           )
    elif configs['dataset']['name'] == 'cityscapes':
        city_mode = 'train'  # Can be trainval
        city_quality = 'fine'
        train_set = args.dataset_cls.CityScapes(
            # ! if not use video prediction method for training data synthesis, then maxSkip = 0
            city_quality, city_mode, 0,
            joint_transform=train_joint_transform,
            transform=train_input_transform,
            target_transform=target_train_transform,
            dump_images=configs['dataset']['dump_aug_img'],
            cv_split=configs['dataset']['cv_split'])
        val_set = args.dataset_cls.CityScapes('fine', 'val', 0,
                                              transform=val_input_transform,
                                              target_transform=target_transform,
                                              cv_split=configs['dataset']['cv_split'])
    else:
        raise Exception('Dataset {} is not supported.'.format(
            configs['dataset']['name']))

    # Set dataloader
    if args.ngpu > 1:
        # Use default DistributedSampler if apex is not used.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    elif args.ngpu == 1:
        train_sampler = None
        val_sampler = None
    args.num_workers = 4 * args.ngpu
    # DDP: shuffle = False, and set_epoch() in each epoch.
    # drop_last: To deal with batch normalization.
    train_loader = DataLoader(train_set, batch_size=configs['training']['batch_size'],
                              num_workers=args.num_workers, shuffle=(
                                  train_sampler is None),
                              drop_last=True, sampler=train_sampler
                              )
    val_loader = DataLoader(val_set, batch_size=configs['training']['val_batch_size'],
                            num_workers=args.num_workers // 2, shuffle=False,
                            drop_last=False, sampler=val_sampler)

    return train_loader, val_loader, train_set
