import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from lib.utils.distributed import is_distributed
from lib.utils.tools.logger import Logger as Log


def _get_list_from_env(name):

    value = os.environ.get(name)
    if value is None:
        return None

    return [x.strip() for x in value.split(',')]


class DataHelper:

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer
        self.conditions = configer.conditions
        self.vis_uncertainty = self.configer.get('uncertainty_visualizer', 'vis_uncertainty')
        self.use_boundary = self.configer.get('protoseg', 'use_boundary')

    def input_keys(self):
        env_value = _get_list_from_env('input_keys')
        if env_value is not None:
            inputs = env_value
        elif self.conditions.use_sw_offset:
            inputs = ['img', 'offsetmap_h', 'offsetmap_w']
        elif self.conditions.use_dt_offset:
            inputs = ['img', 'distance_map', 'angle_map']
        else:
            inputs = ['img']

        return inputs

    def name_keys(self):
        names = ['name']

        return names
    
    def boundary_keys(self):
        boundary_maps = ['boundarymap']
        
        return boundary_maps
        
    def img_keys(self):
        imgs = ['img']
            
        return imgs

    def target_keys(self):

        env_value = _get_list_from_env('target_keys')
        if env_value is not None:
            return env_value
        elif self.conditions.pred_sw_offset:
            targets = [
                'labelmap',
                'offsetmap_h',
                'offsetmap_w',
            ]
        elif self.conditions.pred_dt_offset:
            targets = [
                'labelmap',
                'distance_map',
                'angle_map',
            ]
        elif self.conditions.pred_ml_dt_offset:
            targets = [
                'labelmap',
                'distance_map',
                'multi_label_direction_map',
            ]
        else:
            targets = ['labelmap']

        return targets

    def _reverse_data_dict(self, data_dict):
        result = {}
        for k, x in data_dict.items():

            if not isinstance(x, torch.Tensor):
                result[k] = x
                continue

            new_x = torch.flip(x, [len(x.shape) - 1])

            # since direction_label_map, direction_multilabel_map will not appear in inputs, we omit the flipping
            if k == 'offsetmap_w':
                new_x = -new_x
            elif k == 'angle_map':
                new_x = x.clone()
                mask = (x > 0) & (x < 180)
                new_x[mask] = 180 - x[mask]
                mask = (x < 0) & (x > -180)
                new_x[mask] = - (180 + x[mask])

            result[k] = new_x

        return result

    def _prepare_sequence(self, seq, force_list=False, name_seq=False, img_seq=False):

        def split_and_cuda(
                lst: 'List[List[Tensor, len=N]]', device_ids) -> 'List[List[Tensor], len=N]':
            results = []
            for *items, d in zip(*lst, device_ids):
                if len(items) == 1 and not force_list:
                    results.append(items[0].unsqueeze(0).cuda(d))
                else:
                    results.append([
                        item.unsqueeze(0).cuda(d)
                        for item in items
                    ])
            return results

        if self.conditions.diverse_size and not self.trainer.seg_net.training:

            if is_distributed():
                assert len(seq) == 1
                seq = [x.unsqueeze(0) for x in seq[0]]
                return self.trainer.module_runner.to_device(*seq, force_list=force_list)

            device_ids = list(range(len(self.configer.get('gpu'))))
            return split_and_cuda(seq, device_ids)
        else:
            return self.trainer.module_runner.to_device(
                *seq, force_list=force_list, name_seq=name_seq, img_seq=img_seq)

    def prepare_data(self, data_dict, want_reverse=False):

        input_keys, target_keys = self.input_keys(), self.target_keys()

        if self.vis_uncertainty and self.configer.get('phase') == 'val':
            name_keys = self.name_keys()
            names = [data_dict[k] for k in name_keys]
            img_keys = self.img_keys()
            imgs = [data_dict[k] for k in img_keys]
            # Log.info_once('Image name keys: {}'.format(name_keys))
            # Log.info_once('Image keys: {}'.format(img_keys))
        
        if self.use_boundary and self.configer.get('phase') != 'test':
            boundary_keys = self.boundary_keys()
            boundary_maps = [data_dict[k] for k in boundary_keys]
            Log.info_once('Boundary map keys: {}'.format(boundary_keys))
            

        if self.conditions.use_ground_truth:
            input_keys += target_keys

        Log.info_once('Input keys: {}'.format(input_keys))
        Log.info_once('Target keys: {}'.format(target_keys))

        inputs = [data_dict[k] for k in input_keys]
        batch_size = len(inputs[0])
        targets = [data_dict[k] for k in target_keys]

        if self.vis_uncertainty and self.configer.get('phase') == 'val':
            sequences = [
                self._prepare_sequence(inputs, force_list=True),
                self._prepare_sequence(targets, force_list=False),
                self._prepare_sequence(names, force_list=True, name_seq=True),
                self._prepare_sequence(imgs, force_list=True, img_seq=True)
            ]
        elif self.use_boundary and self.configer.get('phase') != 'test':
            sequences = [
                self._prepare_sequence(inputs, force_list=True),
                self._prepare_sequence(targets, force_list=False),
                self._prepare_sequence(boundary_maps, force_list=False)
            ]
        else:
            sequences = [
                self._prepare_sequence(inputs, force_list=True),
                self._prepare_sequence(targets, force_list=False)
            ]
        if want_reverse:
            rev_data_dict = self._reverse_data_dict(data_dict)
            sequences.extend([
                self._prepare_sequence(
                    [rev_data_dict[k] for k in input_keys],
                    force_list=True
                ),
                self._prepare_sequence(
                    [rev_data_dict[k] for k in target_keys],
                    force_list=False
                )
            ])

        return sequences, batch_size
