from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import shutil

from lib.utils.tools.logger import Logger as Log


class ImgExtractor(object):
    def __init__(self, args):
        super(ImgExtractor, self).__init__()
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.phase = args.phase
        self.img_interval = args.img_interval
        
    def __get_img_list(self, dir_name):
        filename_list = list()
        for item in os.listdir(dir_name):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))
            else: 
                filename_list.append(item)
                
        return filename_list
        
    def __divide_raw_data(self):
        dir_name = os.path.join(self.input_dir, )
        img_list = self.__get_img_list()
    
        for i in range(0, len(img_list), self.img_interval):
            img_name = img_list[i].split('/')[-1]
            #! DO NOT CHANGE TIMESTAMP 
            shutil.copy(img_list[i], os.path.join(self.output_dir, self.phase, img_name))
            
    def extract_img(self):
        self.__divide_raw_data
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data', type=str,
                        dest='input_dir', help='directory of input image.')
    parser.add_argument('--output_dir', default='/data', type=str,
                        dest='output_dir', help='directory of output image.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='sequence name')
    parser.add_argument('--img_interval', default='20', type=int,
                        dest='img_interval', help='interval for annotation')
    parser.add_argument('--start_id', default='20', type=int,
                        dest='img_interval', help='interval for annotation')
    args = parser.parse_args()
    
    img_extractor = ImgExtractor(args)
    
    img_extractor.extract_img()