""" 
Set sampler of the dataloader, and get dataloaders at different phases.
Convert images and labels to GPU tensors.
"""


import torch
from torch.utils import data


from datasets.tools import cv2_aug_transforms
import datasets.tools.transforms as trans
from utils.tools.logger import Logger as Log
from datasets.tools.collate import collate
from utils.distributed import is_distributed, get_world_size, get_rank
from datasets.loader.default_loader import DefaultLoader, DefaultTestLoader


class DataLoader(object):
    def __init__(self, configer):
        self.configer = configer
        # Use cv2
        # Data Augmentations for train/val set
        self.aug_train_transform = cv2_aug_transforms.CV2AugCompose(
            self.configer, split='train')
        self.aug_val_transform = cv2_aug_transforms
        # Transforms for all imgs
        self.img_transform = trans.Compose([
            # Convert a ``numpy.ndarray or Image`` to tensor.
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])
        self.label_transform = trans.Compose([
            trans.ToLabel(),  # to tensor
            trans.ReLabel(255, -1)  # Remap label of background(255) to -1.
        ])

    def get_dataloader_sampler(self, klass, split, dataset):
        """
        klass: DataLoader for the Dataset
        """
        from datasets.loader.multi_dataset_loader import MultiDatasetLoader, MultiDatasetTrainingSampler

        root_dir = self.configer.get('data', 'data_dir')
        if isinstance(root_dir, list) and len(root_dir) == 1:  # single dataset
            root_dir = root_dir[0]

        kwargs = dict(
            dataset=dataset,
            aug_transform=(self.aug_train_transform if split ==
                           'train' else self.aug_val_transform),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer
        )

        if isinstance(root_dir, str):
            # Initialzie the dataloader.
            loader = klass(root_dir, **kwargs)
            multi_dataset = False
        elif isinstance(root_dir, list):  # multiple dataset
            loader = MultiDatasetLoader(root_dir, klass, **kwargs)
            multi_dataset = True
            Log.info('Use multi-dataset for {}.'.format(dataset))
        else:
            raise RuntimeError(
                'Unknown dataset root dir: {}.'.format(root_dir))

        if split == 'train':
            if is_distributed() and multi_dataset:
                raise RuntimeError(
                    'Currently multi dataset doesn\'t support distributed learning.')
            elif multi_dataset:
                sampler = MultiDatasetTrainingSampler(loader)
            else:
                sampler = None  # single dataset & non-distributed learning

        elif split == 'val':
            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    loader)
            else:
                sampler = None

        return loader, sampler

    def get_trainloader(self):
        """ 
        Determines the dataset/dataloader based on the configer.
        train_batch_size is the total of all gpus.
        """
        if self.configer.exists('data', 'loader') and (self.configer.get('train', 'loader') in ['hkustgz', 'cityscapes', 'mapillary']):
            Log.info('Use HKUSTGZLoader for training.')
            klass = DefaultLoader  # Initializate it afterwards.
        loader, sampler = self.get_dataloader_sampler(klass, 'train', 'train')
        trainloader = data.DataLoader(
            loader,
            batch_size=self.configer.get('train', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get(
                'data', 'workers') // get_world_size(),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )

        return trainloader

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset

        if self.configer.get('method') == 'sdc':
            Log.info('Use {} method with DefaultLoader for validation .')
            klass = DefaultLoader
        else:
            Log.error('Method: {} loader is invalid.'.format(
                self.configer.get('method')))

        loader, sampler = self.get_dataloader_sampler(klass, 'val', dataset)
        valloader = data.DataLoader(
            loader,
            sampler=sampler,
            batch_size=self.configer.get('val', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )
        return valloader

    def get_testloader(self, dataset=None):
        dataset = 'test' if dataset is None else dataset
        if self.configer.get('method') == 'sdc':
            Log.info('Use {} for test.'.format(DefaultTestLoader))
            test_loader = data.DataLoader(
                DefaultTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                  img_transform=self.img_transform,
                                  configer=self.configer),
                batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'), shuffle=False,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('test', 'data_transformer')
                )
            )
            return test_loader
        else:
            Log.error('Invalid method: {} for test.'.format(
                self.configer.get('method')))
            raise RuntimeError('Invalid method: {} for test.'.format(
                self.configer.get('method')))
