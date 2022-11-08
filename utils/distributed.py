import torch
import os
import sys
import subprocess

from utils.tools.logger import Logger as Log


def is_distributed():
    return torch.distributed.is_initialized()

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def handle_distributed(args, main_file):
    if not args.distributed:  # single gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            map(str, args.gpu))  # use ',' to concat each gpu str
        return

    if args.local_rank >= 0:
        _setup_process_group(args)
        return

    # Distributed Training
    curr_env = os.environ.copy()
    if curr_env.get('CUDA_VISIBLE_DEVICES') is None:  # Use DDP
        curr_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        world_size = len(args.gpu)
    else:  # Single GPU
        world_size = len(curr_env['CUDA_VISIBLE_DEVICES'].split(','))

    curr_env['WORLD_SIZE'] = str(world_size)
    print('World Size: ', world_size)

    py_exec = sys.executable  # python interpreter
    command_args = sys.argv
    Log.info('{}'.format(command_args))

    try:
        main_ind = command_args.index('main.py')
    except:
        print('main.py not available')

    # get the commands after 'main.py'
    command_args = command_args[main_ind+1:]
    print(command_args)
    # Add DDP args into the original commands
    command_args = [
        py_exec, '-u',  # unbuffered stderr
        '-m', 'torch.distributed.launch',
        '--nproc_per_node', str(world_size),
        main_file
    ] + command_args
    process = subprocess.Popen(command_args, env=curr_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=command_args)
    sys.exit(process.returncode)


def _setup_process_group(args):
    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        # rank=local_rank
    )
