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
import pdb

import numpy as np
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class DefaultBoundaryLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, boundary_label_transform=None,
                 configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.boundary_label_transform = boundary_label_transform
        self.img_list, self.label_list, self.name_list, self.edge_label_list = self.__list_dirs(
            root_dir, dataset)
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

        self.use_boundary = self.configer.get('protoseg', 'use_boundary')

        Log.info('{} {}'.format(dataset, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get(
                                         'data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        # Log.info('{}'.format(self.img_list[index]))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        # 0: non-edge, 255: edge
        boundary_map = ImageHelper.read_image(self.edge_label_list[index], tool=self.configer.get(
            'data', 'image_tool'), mode='P')  # ! 0: non-edge, 255： edge, but contains void class
        assert len(np.unique(boundary_map)) == 2
        
        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)
        if self.configer.exists('data', 'reduce_zero_label'):
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap)
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap, boundary_map = self.aug_transform(
                img, labelmap=labelmap, boundary_map=boundary_map)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)
            # 0:non-edge, 255: edge -> 0: non-edge, 1: edge
            boundary_map = self.boundary_label_transform(boundary_map)
            
        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            boundarymap=DataContainer(boundary_map, stack=self.is_stack),
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
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()
        edge_label_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        edge_label_dir = os.path.join(root_dir, dataset, 'edge')

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        # support the argument to pass the file list used for training/testing
        file_list_txt = os.environ.get('use_file_list')
        files = []
        if file_list_txt is None:
            # files = sorted(os.listdir(image_dir))
            seq_list = os.listdir(image_dir)
            for seq in seq_list:
                seq_dir = os.path.join(image_dir, seq)
                for f in os.listdir(seq_dir):
                    files.append(f)

            files = sorted(files)

        else:
            Log.info("Using file list {} for training".format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                files = [x.strip() for x in f]

        for file_name in files:
            seq_name = file_name.split('_')[0]
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}'.format(seq_name), '{}'.format(file_name))
            # label_path = os.path.join(label_dir, image_name + '.png')
            label_name = image_name.replace(
                'leftImg8bit', 'gtFine_labelIds') + '.png'
            label_path = os.path.join(label_dir, '{}'.format(seq_name), label_name)
            edge_label_name = image_name.replace(
                'leftImg8bit', 'gtFine_color') + '.png'
            edge_label_path = os.path.join(edge_label_dir, '{}'.format(seq_name), edge_label_name)
            
            # Log.info('{} {} {}'.format(image_name, img_path, label_path))
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} {} not exists.'.format(
                    label_path, img_path))
                continue
            if self.configer.get(
                    'protoseg', 'use_boundary') and not os.path.exists(edge_label_path):
                Log.error('Edge Label Path: {} not exists.'.format(edge_label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)
            edge_label_list.append(edge_label_path)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            Log.info("Use validation dataset for training.")
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')
            edge_label_dir = os.path.join(root_dir, 'val/edge')

            # we only use trainval set for training if set include_val
            if self.configer.get('dataset') == 'pascal_voc':
                image_dir = os.path.join(root_dir, 'trainval/image')
                label_dir = os.path.join(root_dir, 'trainval/label')
                edge_label_dir = os.path.join(root_dir, 'trainval/edge')
                img_list.clear()
                label_list.clear()
                name_list.clear()
                edge_label_list.clear()

            if file_list_txt is None:
                files = sorted(os.listdir(image_dir))
            else:
                Log.info("Using file list {} for validation".format(file_list_txt))
                with open(os.path.join(root_dir, 'val', 'file_list', file_list_txt)) as f:
                    files = [x.strip() for x in f]

            for file_name in files:
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}'.format(file_name))
                label_path = os.path.join(label_dir, image_name + '.png')
                edge_label_path = os.path.join(edge_label_dir, image_name + '.png')
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} {} not exists.'.format(
                        label_path, img_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)
                edge_label_list.append(edge_label_path)

        if dataset == 'train' and self.configer.get('data', 'include_coarse'):
            Log.info("Use Coarse labeled dataset for training.")
            image_dir = os.path.join(root_dir, 'coarse/image')
            label_dir = os.path.join(root_dir, 'coarse/label')

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_atr'):
            Log.info("Use ATR dataset for training.")
            image_dir = os.path.join(root_dir, 'atr/image')
            label_dir = os.path.join(root_dir, 'atr/label')

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'only_coarse'):
            Log.info("Only use Coarse labeled dataset for training.")
            image_dir = os.path.join(root_dir, 'coarse/image')
            label_dir = os.path.join(root_dir, 'coarse/label')

            img_list.clear()
            label_list.clear()
            name_list.clear()

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'only_mapillary'):
            Log.info("Only use mapillary labeled dataset for training.")
            image_dir = os.path.join(root_dir, 'mapillary/image')
            label_dir = os.path.join(root_dir, 'mapillary/label')

            img_list.clear()
            label_list.clear()
            name_list.clear()

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{}.{}'.format(image_name, "jpg"))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        return img_list, label_list, name_list, edge_label_list


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
