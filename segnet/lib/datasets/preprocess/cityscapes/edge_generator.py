#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: RainbowSecrete
# Generate the edge files and convert the labels of edge pixels / non-edge pixels to void.

# /root/miniconda3/bin/python generate_edge.py

import os
import cv2
import pdb
import glob

import numpy as np

from PIL import Image
from shutil import copyfile


def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape, dtype=np.uint8)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 255

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 255

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 255

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)

    return edge


def generate_train_val_edge(label_path, edge_path, kernel_size=10):
    for label_file in os.listdir(label_path):
        if not label_file.endswith('color.png'):
            continue
        print(label_file)
        label = np.array(Image.open(os.path.join(label_path, label_file)).convert('P'))
        edge = generate_edge(label, kernel_size)
        
        assert edge.shape[-1] == label.shape[-1]
        assert edge.shape[-2] == label.shape[-2]
        
        im_edge = Image.fromarray(edge, 'P')

        edge_file = label_file.replace('label', 'edge')
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)
        im_edge.save(os.path.join(edge_path, edge_file))

        out_edge = np.array(Image.open(os.path.join(edge_path, edge_file)).convert('P'))


def label_edge2void(label_path, edge_path, dest_label_path):
    '''
    Set the pixels along the edge as void label.
    Used to train the models without supervision on the edge pixels.
    '''
    for label_file in os.listdir(label_path):
        if not label_file.endswith('color.png'):
            continue
        print(label_file)
        edge_file = label_file.replace('label', 'edge')

        label = np.array(Image.open(os.path.join(label_path, label_file)).convert('P'))
        edge = np.array(Image.open(os.path.join(edge_path, edge_file)).convert('P'))

        label[edge == 255] = 255
        label_update = Image.fromarray(label)
        
        if not os.path.exists(dest_label_path):
            os.makedirs(dest_label_path)

        label_update.save(os.path.join(dest_label_path, label_file))


def label_nedge2void(label_path, edge_path, dest_label_path):
    '''
    Set the pixels except the edge as void label.
    Used to evaluate the performance of various models on the edge pixels.
    '''
    for label_file in os.listdir(label_path):
        if not label_file.endswith('color.png'):
            continue
        print(label_file)
        edge_file = label_file.replace('label', 'edge')

        label = np.array(Image.open(os.path.join(label_path, label_file)).convert('P'))
        edge = np.array(Image.open(os.path.join(edge_path, edge_file)).convert('P'))
        #! In the ori_label, the pixels with 255 are still 255 
        label[edge == 0] = 255 # edge: black 0, non-edge: white(void) 255
        label_update = Image.fromarray(label)
        
        if not os.path.exists(dest_label_path):
            os.makedirs(dest_label_path)
        
        label_update.save(os.path.join(dest_label_path, label_file))


def calculate_edge(edge_path):
    '''
    Set the pixels except the edge as void label.
    Used to evaluate the performance of various models on the edge pixels.
    '''
    edge_cnt = 0.0
    non_edge_cnt = 0.0

    print("ratio: {:f}".format(1/2))

    for label_file in os.listdir(label_path):
        print(label_file)
        edge_file = label_file.replace('label', 'edge')
        edge = np.array(Image.open(edge_path + edge_file).convert('P'))

        edge_cnt += np.sum(edge == 255)
        non_edge_cnt += np.sum(edge == 0)

    print("ratio: {:f}".format(edge_cnt/non_edge_cnt))


if __name__ == "__main__":
    label_path = "/data/Cityscapes/train/label/"
    edge_path = "/data/Cityscapes/train/edge/"
    # label_path = "/data/Cityscapes/val/label/"
    # edge_path = "/data/Cityscapes/val/edge/"
    for seq in os.listdir(label_path):
        generate_train_val_edge(os.path.join(label_path, seq), os.path.join(edge_path, seq), 5)
        label_nedge2void_path = "/data/Cityscapes/train/label_non_edge_void/"
        label_nedge2void(os.path.join(label_path, seq), os.path.join(edge_path, seq), os.path.join(label_nedge2void_path, seq))

    # calculate_edge(edge_path)
