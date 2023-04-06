from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os
import shutil
from image_similarity_measures.quality_metrics import fsim, rmse

import sys
sys.path.append('/home/hkustgz_segnet/segnet')
from lib.utils.tools.logger import Logger as Log


class ImgExtractor(object):
    def __init__(self, args, input_dir, output_dir):
        super(ImgExtractor, self).__init__()
        self.args = args
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_interval = args.img_interval
        self.ori_timestamp_dir = args.ori_timestamp_dir
        self.required_timestamps_dir = args.required_timestamps_dir
        self.start_id = args.start_id
        self.txt_output_dir = args.txt_output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else: 
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
        
    def __get_img_list(self, dir_name):
        filename_list = list()
        for item in sorted(os.listdir(dir_name)):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))
            else: 
                filename_list.append(item)
                
        return filename_list
        
    def extract_img_from_txt(self, img_id_file):
        ''' 
        Based on required timestamp to divide raw data into several sequences.
        '''
        img_list = self.__get_img_list(self.input_dir)
        
        with open(img_id_file, 'r') as f: 
            target_inds = f.readlines()
            f.close()
    
        seq_name_list = []
        img_id_list = []
        seq_img_list = []
        for ind in target_inds:
            ind = ind.strip()
            if ind[:8] == 'sequence':
                seq_dir = os.path.join(self.output_dir, '{}'.format(ind))
                
                if os.path.exists(seq_dir):
                    shutil.rmtree(seq_dir)
                os.makedirs(seq_dir)
                
                seq_name_list.append(ind)
                continue
            else: 
                img_id_list.append(int(ind))
        
        for i, seq_name in enumerate(seq_name_list):
            seq_img_list.append(seq_name + '\n')
            start_id = img_id_list[i * 2]
            end_id = img_id_list[i * 2 + 1] - 1
            
            # copy the annotated img to new seq folder
            for img_id in range(start_id, end_id, self.img_interval):
                img_name = img_list[img_id]
                #! DO NOT CHANGE TIMESTAMP 
                shutil.copy(os.path.join(self.input_dir, img_name), os.path.join(self.output_dir, seq_name, img_name))
                seq_img_list.append(str(img_id) + '\n')
            if end_id not in seq_img_list:
                img_name = img_list[end_id]
                shutil.copy(os.path.join(self.input_dir, img_name), os.path.join(self.output_dir, seq_name, img_name))
                seq_img_list.append(str(end_id) + '\n')
                
        # save the corresponding img_id_list to txt file    
        seq_img_id_file = os.path.join(self.txt_output_dir, 'seq_img_id.txt')
        with open(seq_img_id_file, 'w') as f:
            f.writelines(seq_img_list)
        f.close()    
        Log.info('Sequence image id list successfully saved in {}'.format(seq_img_id_file))
            
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
        
    def divide_sequence(self):
        ''' 
        1. get img id according to the required time and the corresponding timestsamp.txt -> get different sequences
        2. divide images by a fixed interval in each sequence
        3. save the extracted image for each sequence.
        '''
        with open(self.ori_timestamp_dir, 'r') as f:
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
        
        img_id_file = os.path.join(self.txt_output_dir, 'seq_img_id.txt')
        with open(img_id_file, 'w') as f:
            f.writelines(target_inds)
        f.close()
        Log.info('Img ID txt successfully saved in {}'.format(img_id_file))
        
        return img_id_file
    
    def extract_img(self):
        img_list = self.__get_img_list(self.input_dir)
        
        start_id = self.start_id
        end_id = len(img_list) - 1
        
        seq_img_list = []
        
        for img_id in range(start_id, end_id, self.img_interval):
            img_name = img_list[img_id]
            #! DO NOT CHANGE TIMESTAMP 
            shutil.copy(os.path.join(self.input_dir, img_name), os.path.join(self.output_dir, img_name))
            seq_img_list.append(str(img_id) + '\n')
            
        # save the corresponding img_id_list to txt file    
        seq_img_id_file = os.path.join(self.txt_output_dir, 'seq_img_id.txt')
        with open(seq_img_id_file, 'w') as f:
            f.writelines(seq_img_list)
        f.close()    
        Log.info('Sequence image id list successfully saved in {}'.format(seq_img_id_file))
        
if __name__ == "__main__":
    input_dir_list = [
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence00/data/frame_cam01/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence01/data/frame_cam00/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence01/data/frame_cam01/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence02/data/frame_cam00/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence02/data/frame_cam01/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence03/data/frame_cam00/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence03/data/frame_cam01/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence04/data/frame_cam00/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence04/data/frame_cam01/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence05/data/frame_cam00/image/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence05/data/frame_cam01/image/data',
]

    output_dir_list = [
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence00/data/frame_cam01/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence01/data/frame_cam00/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence01/data/frame_cam01/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence02/data/frame_cam00/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence02/data/frame_cam01/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence03/data/frame_cam00/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence03/data/frame_cam01/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence04/data/frame_cam00/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence04/data/frame_cam01/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence05/data/frame_cam00/image_annotated/data',
    '/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence05/data/frame_cam01/image_annotated/data',
]

    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_output_dir', default='/home/hkustgz_segnet/segnet/lib/annotation_tools/timestamp', type=str, dest='txt_output_dir', help='directory of timestamp txt output.')
    parser.add_argument('--img_interval', default='100', type=int, dest='img_interval', help='interval for annotation')
    parser.add_argument('--start_id', default='0', type=int, dest='start_id', help='interval for annotation')
    parser.add_argument('--ori_timestamp_dir', default='/data/Data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230309_hkustgz_campus_road_day/data/frame_cam00/image/timestamps.txt', type=str, dest='ori_timestamp_dir', help='original dir of timestamp.txt')
    parser.add_argument('--required_timestamps_dir', default='/home/hkustgz_segnet/segnet/lib/annotation_tools/timestamp/sequence_timestamp.txt', type=str, dest='required_timestamps_dir', help='dir of required timestamps for sequences')
    parser.add_argument('--divide_sequence', help='use provided timestamp to divide sequences',
    default=False, type=bool, dest='divide_sequence')
    
    args = parser.parse_args()
    
    for i, input_dir in enumerate(input_dir_list):
        output_dir = output_dir_list[i]
        img_extractor = ImgExtractor(args, input_dir, output_dir)
        
        img_id_file = None
        
        if args.divide_sequence:
            img_id_file = img_extractor.divide_sequence()
            img_extractor.extract_img_from_txt(img_id_file)
        else: 
            img_extractor.extract_img()