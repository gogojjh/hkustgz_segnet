""" 
Loss Definitions
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: loss
def get_loss(configs, args):
    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, size_average=True,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL(classes=args.dataset_cls.num_classes,
                                     ignore_index=args.dataset_cls.ignore_label,
                                     upper_bound=args.wt_bound).cuda()
    else:
        criterion = CrossEntropyLoss2d(size_average=True,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    criterion_val = CrossEntropyLoss2d(size_average=True,
                                       weight=None,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    return criterion, criterion_val
