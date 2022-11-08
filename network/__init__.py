"""
Network Initializations
"""

import logging
import torch.nn as nn
import importlib


def get_model(network, num_classes, criterion):
    """ 
    Fetch network function pointer.
    arch example: 'network.deepv3.DeepWV3Plus' (network.module.model)
    """
    module = network[:network.rfind('.')]  # Get network framework (deepv3).
    model = network[network.rfind('.') + 1:]  # Get model type.
    # Import user-defined module. (e.g., deepv3.py)
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)  # Get network attribute of this module.
    # Construct a net object.
    net = net_func(num_classes=num_classes, criterion=criterion)

    return net


def get_net(configs, args, criterion):
    """ 
    Get Network Architecture based on arguments provided.
    """
    net = get_model(network=configs['model']['arch'],
                    num_classes=args.dataset_cls.num_classes, criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.fomat(num_params / 1000000))

    net = net.cuda()

    return net


def wrap_net_in_ddp(net, args):
    """ 
    Wrap the network in DistributedDataParallel.
    """
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank], output_device=args.local_rank)

    return net
