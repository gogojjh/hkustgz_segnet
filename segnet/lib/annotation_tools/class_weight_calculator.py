import os
import sys
import shutil
import numpy as np
import cv2
sys.path.append('/home/hkustgz_segnet/segnet')
from lib.utils.helpers.image_helper import ImageHelper


LABEL_LIST = np.array([
			10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
			115, 120, 125, 130
		])
COLOR_LABEL_LIST = np.array([
    [0, 162, 170],
    [162, 248, 63],
    [241, 241, 241],
    [147, 109, 6],
    [122, 172, 9],
    [12, 217, 219],
    [7, 223, 115],
    [161, 161, 149],
    [231, 130, 130],
    [252, 117, 35],
    [92, 130, 216],
    [0, 108, 114],
    [0, 68, 213],
    [91, 0, 231],
    [227, 68, 255],
    [168, 38, 191],
    [106, 0, 124],
    [255, 215, 73],
    [209, 183, 91],
    [244, 255, 152],
    [138, 164, 165],
    [175, 0, 106],
    [228, 0, 140],
    [234, 178, 200],
    [255, 172, 172]
])
IGNORE_LABEL_ID = 152,
IGNORE_COLOR_LABEL_ID = np.array([177, 165, 25])


class ClassWeightCalculator(object):
    ''' 
    Calculate class weights of cross-entropy loss using label frequency.
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.image_helper = ImageHelper()
        self.class_freq_arr = np.zeros((num_classes,), dtype=np.uint64)
        # self.class_freq_arr = np.array([629242445, 
        #                                 134025205,  
        #                                 14316102,  
        #                                 30786196,  
        #                                 74103937,  
        #                                 12556419,  
        #                                 17420688,
        #                                 13497554,   
        #                                 2015437,    
        #                                 179249,   
        #                                 7580802,    
        #                                 174064,   
        #                                 903274,   
        #                                 2098901,
        #                                 715407552,
        #                                 20629326,   
        #                                 8681549, 
        #                                 436461082,
        #                                 335441037,
        #                                 4745801,
        #                                 49691941,
        #                                 3280369,
        #                                 4303,
        #                                 18782216,
        #                                 624798887])
        self.class_weight_arr = np.zeros((num_classes,), dtype=np.uint64)
        
        
    def calc_class_freq(self, root_dir, img_type):
        for seq_name in os.listdir(os.path.join(root_dir, img_type)):
            for frame_name in os.listdir(os.path.join(root_dir, img_type, seq_name)):
                for f in os.listdir(os.path.join(root_dir, img_type, seq_name, frame_name)):
                    img_path = os.path.join(root_dir, img_type, seq_name, frame_name, f)
                    if img_type == 'label_id':
                        label_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        for i in range(len(LABEL_LIST)):
                            class_id = LABEL_LIST[i]
                            class_freq = np.sum(label_img == class_id)
                            self.class_freq_arr[i] += class_freq   

                    elif img_type == 'label_color':
                        label_img = self.image_helper.cv2_read_image(img_path, mode='RGB')
                        
                        for i in range(len(COLOR_LABEL_LIST)):
                            class_id = COLOR_LABEL_LIST[i]
                            class_freq = np.sum(np.all(label_img == class_id, axis=-1))
                            self.class_freq_arr[i] += class_freq   
                            
                    print('Freq calculation finished: {}'.format(img_path))
                            
        print   ('class_freq_arr: {}'.format(self.class_freq_arr))
        
    def calc_class_weight(self):
        freq_sum = np.sum(self.class_freq_arr)
        freq_max = np.max(self.class_freq_arr)
        for i in range(self.num_classes):
            self.class_weight_arr[i] = freq_sum / self.class_freq_arr[i]
            # self.class_weight_arr[i] = 1 / self.class_freq_arr[i]
            
                            
        print('class_weight_arr: {}'.format(self.class_weight_arr))
        
        
''' 
class_freq_arr: [629242445 134025205  14316102  30786196  74103937  12556419  17420688
  13497554   2015437    179249   7580802    174064    903274   2098901
 715407552  20629326   8681549 436461082 335441037   4745801  49691941
   3280369      4303  18782216 624798887]
'''
        

if __name__ == "__main__":
    cls_weihgt_calculator = ClassWeightCalculator(25)
    cls_weihgt_calculator.calc_class_freq('/data/HKUSTGZ/train', 'label_id')
    cls_weihgt_calculator.calc_class_weight()
        
        

                

