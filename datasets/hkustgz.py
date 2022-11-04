""" 
HKUSTGZ Dataset Loader
Label format follows CityScapes dataset
Different datasets refer to different data augmentation/training data synthesis methods.
"""
import logging
import json
import os
import numpy as np
from PIL import Image
from torch.utils import data
from ruamel.yaml import YAML

import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.hkustgz_labels as hkustgz_labels

from config import cfg


trainid_to_name = hkustgz_labels.trainId2name
id_to_trainid = hkustgz_labels.label2trainid
num_classes = 19
ignore_label = 255  # Define the ignore labels for this dataset.
root = cfg.DATASET.HKUSTGZ_DIR
aug_root = cfg.DATASET.HKUSTGZ_AUG_DIR

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


# TODO Consider other methods to augment the training set.
def add_items(items, aug_items, routes, img_path, mask_path, mask_postfix, mode, maxSkip):
    """ 
    Add More items to the list of the augmented dataset
    For training set, uniform sample from the augmented extra dataset with an interval of maxSkip, and also return aug_items.
    Augmented samples for training follow the joint propagation method described in "Improving Semantic Segmentation via Video Propagation and Label Relaxation."
    """
    for r in routes:
        r_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(
            os.path.join(img_path, r))]  # list of all img num in this route
        for it in r_items:
            item = (os.path.join(img_path, r, it + '_leftImg8bit.png'),
                    os.path.join(mask_path, r, it + mask_postfix))
            ########################################################
            ###### dataset augmentation for training ###############
            ########################################################
            if mode == 'train' and maxSkip > 0:
                new_img_path = os.path.join(
                    aug_root,  'leftImg8bit_trainvaltest', 'leftImg8bit')
                new_mask_path = os.path.join(
                    aug_root, 'gtFine_trainvaltest', 'gtFine')
                file_info = it.split('_')
                cur_seq_id = file_info[-1]

                # use 0 for padding if less than 6 letters
                # select nearby frames defined by maxSkip
                prev_seq_id = "%06d" % (int(cur_seq_id) - maxSkip)
                next_seq_id = "%06d" % (int(cur_seq_id) + maxSkip)
                # naming policy: 'aachen(place/route)_000000(frame num)_000019(unknown)_leftImg8bit.png'
                prev_it = file_info[0] + '_' + file_info[1] + '_' + prev_seq_id
                next_it = file_info[0] + '_' + file_info[1] + "_" + next_seq_id
                prev_item = (os.path.join(new_img_path, r, prev_it + '_leftImg8bit.png'),
                             os.path.join(new_mask_path, r, prev_it + mask_postfix))
                if os.path.isfile(prev_item[0]) and os.path.isfile(prev_item[1]):
                    aug_items.append(prev_item)
                next_item = (os.path.join(new_img_path, r, next_it + '_leftImg8bit.png'),
                             os.path.join(new_mask_path, r, next_it + mask_postfix))
                if os.path.isfile(next_item[0]) and os.path.isfile(next_item[1]):
                    aug_items.append(next_item)
            items.append(item)


def make_cv_splits(img_dir_name):
    """ 
    Create splits of train/val data.
    A split is a list of routes.
    split0 is aligned with the default HKUSTGZ train/val.
    """
    trn_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'train')
    val_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'val')

    trn_routes = ['train/' + r for r in os.listdir(trn_path)]
    val_routes = ['val/' + r for r in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_routes = sorted(trn_routes)

    all_routes = val_routes + trn_routes
    num_val_routes = len(val_routes)
    num_routes = len(all_routes)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_routes // cfg.DATASET.CV_SPLITS
        for j in range(num_routes):
            if j >= offset and j < (offset + num_val_routes):
                split['val'].append(all_routes[j])
            else:
                split['train'].append(all_routes[j])
        cv_splits.append(split)

    return cv_splits


def make_split_coarse(img_path):
    """ 
    Create a train/val split for coarse
    """
    all_routes = os.listdir(img_path)
    # ascending order, needs to always be the same
    all_routes = sorted(all_routes)
    val_routes = []  # Can manually set routes to not be included into train split

    split = {}
    split['val'] = val_routes
    split['train'] = [r for r in all_routes if r not in val_routes]
    return split


def make_test_split(img_dir_name):
    test_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'test')
    test_routes = ['test/' + r for r in os.listdir(test_path)]

    return test_routes


def make_dataset(quality, mode, maxSkip=0, fine_coarse_mult=6, cv_split=0):
    """
    Assemble list of images + mask files, return default dataset and augmented dataset

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    aug_items = []

    if quality == 'coarse':
        assert (cv_split == 0)
        assert mode in ['train', 'val']
        img_dir_name = 'leftImg8bit_trainextra'
        img_path = os.path.join(
            root, img_dir_name, 'leftImg8bit', 'train_extra')
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', 'train_extra')
        mask_postfix = '_gtCoarse_labelIds.png'
        coarse_dirs = make_split_coarse(img_path)
        logging.info('{} coarse routes: '.format(
            mode) + str(coarse_dirs[mode]))
        add_items(items, aug_items,
                  coarse_dirs[mode], img_path, mask_path, mask_postfix, mode, maxSkip)
    elif quality == 'fine':
        assert mode in ['train', 'val', 'test', 'trainval']
        img_dir_name = 'leftImg8bit_trainvaltest'
        img_path = os.path.join(root, img_dir_name, 'leftImg8bit')
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine')
        mask_postfix = '_gtFine_labelIds.png'
        cv_splits = make_cv_splits(img_dir_name)
        if mode == 'trainval':
            modes = ['train', 'val']
        else:
            modes = [mode]
        for mode in modes:
            if mode == 'test':
                cv_splits = make_test_split[img_dir_name]
                add_items(items, aug_items, cv_splits, img_path,
                          mask_path, mask_postfix, mode, maxSkip)
            else:
                logging.info('{} fine routes: '.format(
                    mode) + str(cv_splits[cv_split][mode]))
                add_items(items, aug_items, cv_splits[cv_split][mode],
                          img_path, mask_path, mask_postfix, mode, maxSkip)

    else:
        raise 'unknown quality {} for HKUSTGZ dataset.'.format(quality)
    logging.info('HKUSTGZ-{}: {} images'.format(mode,
                 len(items) + len(aug_items)))

    return items, aug_items


""" 
Default HKUSTGZ dataset without uniform sampling.

params:
joint_transform: geometric image transformations
transform: normalize transformation
target_transform: 
"""


class HKUSTGZ(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform=None, transform=None, target_transform=None, dump_images=False, cv_split=None, eval_mode=False, eval_scales=None, eval_flip=False):
        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        if eval_scales != None:
            self.eval_scales = [float(scale)
                                for scale in eval_scales.split(',')]

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'Expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS
                )
        else:
            self.cv_split = 0
        self.imgs, _ = make_dataset(
            quality, mode, self.maxSkip, cv_split=self.cv_split)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the dataset.')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # data augmentation for evaluation
    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(
                    *self.mean_std)(tensor_img)  # params are tuple
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        # basename: tail part of the path
        # splitext: split the file and extension names
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)  # Image.open gets image object, not array
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v  # get array of trainID

        if self.eval_mode:  # eval_mode: do img transform
            return [transforms.ToTensor()(img)], self._eval_get_item(img, mask_copy,
                                                                     self.eval_scales,
                                                                     self.eval_flip),

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image transformations
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs)


# TODO HKUSTGZUniform which uses video prediction model for dataset augmentation
