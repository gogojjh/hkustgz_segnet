from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import shutil

import sys
sys.path.append('/home/hkustgz_segnet/segnet')
from lib.utils.tools.logger import Logger as Log


class ImgExtractor(object):
    def __init__(self, args):
        super(ImgExtractor, self).__init__()
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.phase = args.phase
        self.img_interval = args.img_interval
        self.timestamp_dir = args.timestamp_dir
        self.required_timestamps_dir = args.required_timestamps_dir
        self.output_timestamp_dir = os.path.join(self.output_dir, 'required_t.txt')
        
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
        dir_name = os.path.join(self.input_dir,  )
        img_list = self.__get_img_list()
    
        for i in range(0, len(img_list), self.img_interval):
            img_name = img_list[i].split('/')[-1]
            #! DO NOT CHANGE TIMESTAMP 
            shutil.copy(img_list[i], os.path.join(self.output_dir, self.phase, img_name))
            
    def __match_img_timestamp(self, target_t, timestamps):
        ''' 
        Use nearest timestamp in rosbag to find the corresponding img id (row num in timestamp.txt).
        '''
        target_t = target_t.strip()
        near_timestamps = []
        near_img_id = []
        closest_t = None
        for i, t in enumerate(timestamps):
            t = t.strip()
            if target_t.split('.')[0] != t.split('.')[0]:
                continue
            else: 
                if target_t.split('.')[1][:7] != t.split('.')[1][:7]:
                    near_timestamps.append(int(t.split('.')[-1]))
                    near_img_id.append(str(i))
                else: 
                    closest_t = str(i)
                    break
            
        if closest_t is not None: 
            return closest_t
        else: 
            assert len(near_timestamps) != 0        
            find_closest_t = lambda num, collection:min(collection, key=lambda x:abs(x - num)) 
            closest_t = find_closest_t(int(target_t.split('.')[-1]), near_timestamps)
            
            return near_img_id[near_timestamps.index(closest_t)]
        
    
    def extract_img(self):
        ''' 
        1. get img id according to the required time and the corresponding timestsamp.txt -> get different sequences
        2. divide images by a fixed interval in each sequence
        3. save the extracted image for each sequence.
        '''
        with open(self.timestamp_dir, 'r') as f:
            timestamps = f.readlines()
        f.close()
        
        target_inds = []
        with open(self.required_timestamps_dir, 'r') as f: 
            required_t = f.readlines()
            f.close()
        for i, target_t in enumerate(required_t):
            if str(target_t)[:8] == 'sequence':
                target_inds.append(target_t.strip() + '\n')
            else:
                target_i = self.__match_img_timestamp(target_t, sorted(timestamps))
                target_inds.append(target_i + '\n')
        
        img_id_file = os.path.join(self.output_dir, 'seq_img_id.txt')
        with open(img_id_file, 'w') as f:
            f.writelines(target_inds)
        f.close()
        Log.info('Img ID txt successfully saved in {}'.format(img_id_file))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data', type=str,
                        dest='input_dir', help='directory of input image.')
    parser.add_argument('--output_dir', default='/home/hkustgz_segnet/segnet/lib/annotation_tools/timestamp', type=str,
                        dest='output_dir', help='directory of output image.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='sequence name')
    parser.add_argument('--img_interval', default='20', type=int,
                        dest='img_interval', help='interval for annotation')
    parser.add_argument('--start_id', default='40', type=int,
                        dest='start_id', help='interval for annotation')
    parser.add_argument('--timestamp_dir', default='/data/Data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230309_hkustgz_campus_road_day/data/frame_cam00/image/timestamps.txt', type=str,
                        dest='timestamp_dir', help='dir of timestamp.txt')
    parser.add_argument('--required_timestamps_dir', default='/home/hkustgz_segnet/segnet/lib/annotation_tools/timestamp/sequence_timestamp.txt', type=str,
                        dest='required_timestamps_dir', help='dir of required timestamps for sequences')
    args = parser.parse_args()
    
    img_extractor = ImgExtractor(args)
    
    img_extractor.extract_img()