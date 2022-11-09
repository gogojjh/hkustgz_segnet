from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from torch.utils import data

from utils.tools.logger import Logger as Log
from extensions.parallel.data_container import DataContainer
from utils.helpers.image_helper import ImageHelper


class DefaultLoader(data.Dataset):
    """ 
    Dataloaders for HKUSTGZ, CityScapes, Mapillary Datasets
    """

    def __init__(self, root_dir, aug_transform=None, dataset=None, img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        # Get the lists of available image, label and image names.
        self.img_list, self.label_list, self.name_list = self.__list_dirs(
            root_dir, dataset)
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

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
                                          tool=self.configer.get(
                                              'data', 'image_tool'),
                                          mode='P')
        # Remap original cls ids to [0, num_cls-1]
        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)

        if self.configer.exists('data', 'reduce_zero_label'):
            labelmap = self._reduce_zero_label(labelmap)

        # In case the labelmap is PIL Image
        ori_target = ImageHelper.tonp(labelmap)
        # Remap cls 255 (background cls) to class -1
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap = self.aug_transform(img, labelmap)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)

        border_size = ImageHelper.get_size(img)  # augmented img size

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )

        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(
                self.name_list[index], stack=False, cpu_only=True)
        )

    def _reduce_zero_label(self, labelmap):
        """ 
        To ignore class 0 during training, all cls ids are deducted by 1.
        """
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        """ 
        Remap the original class ids to [0, num_class - 1] for training.
        """
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.float32) * 255  # background label
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[
                i]  # self-defined class id
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(
                encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        name_list = list()  # img name list
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')

        # Get extension of img (png/img).
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        # Add support for passing the data file list from terminal inputs.
        file_list_txt = os.environ.get('use_file_list')
        if file_list_txt is None:
            files = sorted(os.listdir(image_dir))
        else:
            Log.info('Use file list {} for training'.format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                # Remove '0' both in the front and at the end of x.
                files = [x.strip() for x in f]
        for file_name in files:
            # Split file_name using '.'. Then concatenate the splitted seq use '.' except for the last element in the seq.
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}'.format(file_name))
            label_path = os.path.join(label_dir, image_name + '.png')
            # Log.info('{} {} {}'.format(image_name, img_path, label_path))
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label path: {} {} does not exist.'.format(
                    label_path, img_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            name_list.append(image_name)

        # For cityscapes dataset to include coarse data into the training phase.
        if dataset == 'train' and self.configer.get('data', 'include_coarse'):
            Log.info('Include coarse labelled data into the training phase.')
            image_dir = os.path.join(root_dir, 'coarse/image')
            label_dir = os.path.join(root_dir, 'coarse/label')

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{} {}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error(
                        'Label path: {} does not exist.'.format(label_path))
                    continue
                # Add coarse-annotated data to the orignal fine-anntoated data list.
                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'only_coarse'):
            Log.info('Only use coarse-anntoated data for training.')
            image_dir = os.path.join(root_dir, 'coarse/image')
            label_dir = os.path.join(root_dir, 'coarse/label')
            # Clear the fine-anntoated data lists.
            img_list.clear()
            label_list.clear()
            name_list.clear()

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(
                    image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error(
                        'Label Path: {} does not exist.'.format(label_path))
                    continue

        if dataset == 'train' and self.configer.get('data', 'only_mapillary'):
            Log.info("Only use the labeled mapillary dataset for training.")
            image_dir = os.path.join(root_dir, 'mapillary/image')
            label_dir = os.path.join(root_dir, 'mapillary/label')

            img_list.clear()
            label_list.clear()
            name_list.clear()

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(
                    image_name, "jpg"))  # ended with 'jpg', not 'png'
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                name_list.append(image_name)

        return img_list, label_list, name_list


class DefaultTestLoader(data.Dataset):
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
