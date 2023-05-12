# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: JingyiXie, LangHuang, DonnyYou, RainbowSecret
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2019
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from torch.utils import data
import cv2

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class SemanticFusionPortableLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.use_color_label = self.configer.get('data', 'use_color_label')
        self.img_list, self.label_list, self.name_list = self.__list_dirs(
            root_dir, dataset)
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'
        Log.info('{} {}'.format(dataset, len(self.img_list)))
        self.ignore_label_id = self.configer.get('data', 'ignore_label_id')
        self.unlabelled_label_id = self.configer.get('data', 'unlabelled_label_id')
        self.label_id_list = self.configer.get('data', 'label_list')
        self.full_label_list = np.append(self.label_id_list, self.ignore_label_id)
        self.full_label_list = np.append(self.full_label_list, self.unlabelled_label_id)
        
        if self.use_color_label:
            self.color_label_id_list = self.configer.get('data', 'color_label_list')
            self.ignore_color_label_id = self.configer.get('data', 'ignore_color_label_id')
            self.color_full_label_list = np.append(self.color_label_id_list, self.ignore_color_label_id)
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get(
                                         'data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        # Log.info('{}'.format(self.img_list[index])) 
        img_size = ImageHelper.get_size(img)
        if not self.use_color_label:
            labelmap = cv2.imread(self.label_list[index], cv2.IMREAD_GRAYSCALE)

            #todo debug
            # if len(np.unique(np.isin(np.array(labelmap), self.full_label_list))) != 1 or not np.all(np.unique(np.isin(np.array(labelmap), self.full_label_list)) == True):
            #     Log.error('{}'.format(self.label_list[index])) 
            
            assert len(np.unique(np.isin(np.array(labelmap), self.full_label_list))) == 1 and np.all(np.unique(np.isin(np.array(labelmap), self.full_label_list)) == True)
            
            if self.configer.exists('data', 'label_list'):
                labelmap = self._encode_label(labelmap)
        else: 
            labelmap = ImageHelper.read_image(self.label_list[index],
                                            tool=self.configer.get('data', 'image_tool'), mode='RGB')

            if self.configer.exists('data', 'color_label_list'):
                labelmap = self._encode_color_label(labelmap)
            
        if self.configer.exists('data', 'reduce_zero_label'):
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap)
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap=labelmap)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        
        # if np.unique(labelmap).size == 1:
        #     Log.info(self.label_list[index])
        
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(
                self.name_list[index], stack=False, cpu_only=True),
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.label_id_list)):
            class_id = self.label_id_list[i]
            encoded_labelmap[labelmap == class_id] = i
            
        encoded_labelmap[labelmap == self.ignore_label_id] = 255
        #! convert 'unlabelled' to 'void'
        
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap
    
    def _encode_color_label(self, labelmap):
        labelmap = np.array(labelmap) # [1536, 2048, 3]
        
        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.float32) * 255
        
        for i in range(len(self.color_label_id_list)):
            class_id = self.color_label_id_list[i]
            encoded_labelmap[np.all(labelmap == class_id, axis=-1)] = i
        
        encoded_labelmap[np.all(labelmap == self.ignore_color_label_id, axis=-1)] = 255
        
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        if not self.use_color_label:
            label_dir = os.path.join(root_dir, dataset, 'label_id')
        else: 
            label_dir = os.path.join(root_dir, dataset, 'label_color')

        # support the argument to pass the file list used for training/testing
        file_list_txt = os.environ.get('use_file_list')
        files = []
        
        #! /data/HKUSTGZ/train/image/20230403_hkustgz_campus_road_day_sequence00/frame_cam01
        if file_list_txt is None:
            seq_list = os.listdir(image_dir)
            for seq_dir in seq_list:                     
                for frame_dir in os.listdir(os.path.join(image_dir, seq_dir)):
                    for f in os.listdir(os.path.join(image_dir, seq_dir, frame_dir)):
                        img_path = os.path.join(image_dir, seq_dir, frame_dir, f)
                        image_name = '.'.join(f.split('.')[:-1])
                        if not self.use_color_label:
                            label_path = os.path.join(label_dir, seq_dir, frame_dir, image_name + '_gt_id.png')
                        else: 
                            label_path = os.path.join(label_dir, seq_dir, frame_dir, image_name + '_gt_color.png')
                        
                        if not os.path.exists(label_path) or not os.path.exists(img_path):
                            Log.error('Label Path: {} {} not exists.'.format(
                            label_path, img_path))
                            
                            continue
                    
                        img_list.append(img_path)
                        label_list.append(label_path)
                        name_list.append(image_name)
                    
            files = sorted(files)

        else:
            Log.info("Using file list {} for training".format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                files = [x.strip() for x in f]

        return img_list, label_list, name_list


class CSDataTestLoader(data.Dataset):
    def __init__(self, root_dir, dataset=None, img_transform=None, configer=None):
        self.configer = configer
        self.img_transform = img_transform
        self.img_list, self.name_list, self.subfolder_list = self.__list_dirs(
            root_dir, dataset)

        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = (size_mode != 'diverse_size')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get(
                                         'data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        if self.img_transform is not None:
            img = self.img_transform(img)
        meta = dict(
            ori_img_size=img_size,
            border_size=img_size,
        )
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(
                self.name_list[index], stack=False, cpu_only=True),
            subfolder=DataContainer(
                self.subfolder_list[index], stack=False, cpu_only=True),
        )

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        name_list = list()
        subfolder_list = list()
        image_dir = os.path.join(root_dir, dataset)
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        if self.configer.get('dataset') == 'cityscapes' or self.configer.get('dataset') == 'camvid' or \
                self.configer.get('dataset') == 'autonue21':
            for item in os.listdir(image_dir):
                sub_image_dir = os.path.join(image_dir, item)
                for file_name in os.listdir(sub_image_dir):
                    image_name = file_name.split('.')[0]
                    img_path = os.path.join(sub_image_dir, file_name)
                    if not os.path.exists(img_path):
                        Log.error(
                            'Image Path: {} not exists.'.format(img_path))
                        continue
                    img_list.append(img_path)
                    name_list.append(image_name)
                    subfolder_list.append(item)
        else:
            for file_name in os.listdir(image_dir):
                image_name = file_name.split('.')[0]
                img_path = os.path.join(image_dir, file_name)
                if not os.path.exists(img_path):
                    Log.error('Image Path: {} not exists.'.format(img_path))
                    continue
                img_list.append(img_path)
                name_list.append(image_name)
                subfolder_list.append('')

        return img_list, name_list, subfolder_list


if __name__ == "__main__":
    # Test cityscapes loader.
    pass