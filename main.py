from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn

from utils.tools.configer import Configer
from utils.tools.logger import Logger as Log


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  # lower(): Convert capital letters to lower letters.
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='train/val/test phase')
    # nargs='+': 1 or multiple inputs
    parser.add_argument(
        '--gpu', default=[0], nargs='+', type=int, dest='gpu', help='The list of gpus used.')

    # ***********  Data Params  **********
    parser.add_argument('--data_dir', default=None, type=str, nargs='+',
                        dest='data:data_dir', help='Data Directory')
    # include-coarse is only provided for Cityscapes.
    parser.add_argument('--include_coarse', type=str2bool, nargs='?', default=False,
                        dest='data:include_coarse', help='Include coarse-labeled set for training.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train:batch_size', help='Training batch size')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val:batch_size', help='Validation Batch Size')

    # ***********  Training Params  **********
    parser.add_argument('--wandb', default='disabled', type=str,
                        dest='wandb', choices=['disabled', 'online', 'offline'], help='Wandb Activation')

    # ***********  Checkpoint Params  **********
    parser.add_argument('--checkpoints_root', default=None, type=str,
                        dest='checkpoints:checkpoints_root', help='Root Dir for Saving the Model')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='checkpoints:checkpoints_name', help='Checkpoint Name')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='checkpoints:save_iters', help='Iteration Number for Saving the Model')
    parser.add_argument('--save_epoch', default=None, type=int,
                        dest='checkpoints:save_epoch', help='Epoch Number for Saving the Model')

    # ***********  Model Params  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='Model Name')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='Base Network of the Model.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='Pretrained Model Path')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='Checkpoint Path.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match the keys or not.')

    # ***********  Solver Params  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='optim:optim_method', help='Optimizer Method')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='lr:base_lr', help='Learning Rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='lr:lr_policy', help='Learning Rate Policy')

    # ***********   Display Params  **********
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver:max_epoch', help='Max Epoch for Training')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver:max_iters', help='Max Iteration for Training')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver:display_iter', help='Iteration for Displaying the Training Logs')

    # ***********  Logs Params  **********
    parser.add_argument('--logfile_level', default=None, type=str,
                        dest='logging:logfile_level', help='Log Level in Files')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='logging:log_file', help='Logs Path')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        dest='logging:log_to_file', help='Whether to write logging into files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='logging:stdout_level', help='stdout Level for Printing')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=True,
                        dest='logging:rewrite', help='Whether to rewrite files or not.')

    # ***********  Distributed Training Params  **********
    parser.add_argument('--local_rank', type=int, default=-1,
                        dest='local_rank', help='Local Rank of Current Process')
    parser.add_argument('--distributed', action='store_true',
                        dest='distributed', help='Use Multi-processing for Training.')

    args = parser.parse_args()

    # Set up distributed training.
    from utils.distributed import handle_distributed
    handle_distributed(args, os.path.expanduser(os.path.abs(__file__)))

    # Use benchmark mode for speed-up
    cudnn.enabled = True
    cudnn.benchmark = args.cudnn

    # ***********  Set up configer and logger.  **********
    # Update absolute data dir
    configer = Configer(args_parser=args)
    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)

    # Get path of this script and add it to the configer's dict.
    # Get file path of this script.
    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    # Set log file name and update it in the configer's dict.
    if configer.get('logging,' 'log_to_file'):
        log_file = configer.get('logging', 'log_to_file')
        new_log_file = '{}_{}'.format(
            log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)
    # Initialize the logger after get logger-related configs.
    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    Log.info('Batch Size: {}'.format(configer.get('train', 'batch_size')))

    # ***********  Set up method and model.  **********
    model = None
    # Choose the corresponding trainer based on the method set in the config.
    if configer.get('method') == 'sdcnet':
        if configer.get('phase') == 'train':
            from segmentor.trainer.trainer import Trainer
            model = Trainer(configer)
        # todo: tester
        # elif configer.get('phase') == 'test':
    else:
        Log.error('Method: {} is not valid.'.format(configer.get('task')))
        exit(1)
    
    if configer.get('phase') == 'train':
        model.train()
    # todo: test()
    # elif configer.get('phase').startwith('test') and configer.get('network', 'resume') is not None:
        
    else: 
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)
    