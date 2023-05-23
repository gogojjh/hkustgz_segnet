import os
import sys
import shutil


class LabelProcessor(object):
    ''' 
    Select the last few images of each sequence as val sequence.
    '''
    def __init__(self):
        super(LabelProcessor, self).__init__()
        
    def rename_img(self, root_dir, img_type):
        """
        Add seq info & frame00/01 info to img name.
        HKUSTGZ/train/image/20230322_hkustgz_campus_road_day_sequence01/frame_cam01/002500.png
        """
        
        for seq_name in os.listdir(os.path.join(root_dir, img_type)):
            for frame_name in os.listdir(os.path.join(root_dir, img_type, seq_name)):
                for f in os.listdir(os.path.join(root_dir, img_type, seq_name, frame_name)):
                    
                    seq_date = seq_name.split('_')[0]
                    seq_num = seq_name.split('_')[-1]
                    
                    if seq_name.startswith('20230403'):
                        img_num = (f.split('.')[0]).split('_')[-1]
                    else: 
                        img_num = f.split('.')[0]
                        
                    if img_type == 'image':
                        img_name = seq_date + '_' + seq_num + '_' + frame_name + '_' + img_num + '.png'
                    elif img_type == 'label_id': 
                        img_name = seq_date + '_' + seq_num + '_' + frame_name + '_' + img_num + '_gt_id.png'
                    elif img_type == 'label_color': 
                        img_name = seq_date + '_' + seq_num + '_' + frame_name + '_' + img_num + '_gt_color.png'
                    else: 
                        raise RuntimeError('Invalid img type.')
                    
                    shutil.move(os.path.join(root_dir, img_type, seq_name, frame_name, f), os.path.join(os.path.join(root_dir, img_type, seq_name, frame_name, img_name)))
        
    def select_val_seq(self, root_dir):
        ''' 
        select the last few imgs of each seq as part of val seq.
        '''                    
        for seq_name in os.listdir(os.path.join(root_dir, 'train', 'image')):
            for frame_name in os.listdir(os.path.join(root_dir, 'train', 'image', seq_name)):
                val_img_list = sorted(os.listdir(os.path.join(root_dir, 'train', 'image', seq_name, frame_name)))[-2:]
                for img in val_img_list:
                    # move it to val dir
                    if not os.path.exists(os.path.join(root_dir, 'val', 'image', seq_name, frame_name)):
                        os.makedirs(os.path.join(root_dir, 'val', 'image', seq_name, frame_name))
                    shutil.move(os.path.join(root_dir, 'train', 'image', seq_name, frame_name, img), os.path.join(root_dir, 'val', 'image', seq_name, frame_name, img))
                    
                    label_img = img.split('.')[0] + '_gt_id.png'
                    if not os.path.exists(os.path.join(root_dir, 'val', 'label_id', seq_name, frame_name)):
                        os.makedirs(os.path.join(root_dir, 'val', 'label_id', seq_name, frame_name))
                    shutil.move(os.path.join(root_dir, 'train', 'label_id', seq_name, frame_name, label_img), os.path.join(root_dir, 'val', 'label_id', seq_name, frame_name, label_img))
                    
                    color_label_img = img.split('.')[0] + '_gt_color.png'
                    if not os.path.exists(os.path.join(root_dir, 'val', 'label_color', seq_name, frame_name)):
                        os.makedirs(os.path.join(root_dir, 'val', 'label_color', seq_name, frame_name))
                    shutil.move(os.path.join(root_dir, 'train', 'label_color', seq_name, frame_name, color_label_img), os.path.join(root_dir, 'val', 'label_color', seq_name, frame_name, color_label_img))
        
    
if __name__ == "__main__":
    label_processor = LabelProcessor()
    
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'train'), 'image')
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'val'), 'image')
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'train'), 'label_id')
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'val'), 'label_id')
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'train'), 'label_color')
    # label_processor.rename_img(os.path.join('/data/HKUSTGZ', 'val'), 'label_color')
                
    label_processor.select_val_seq('/data/kin/ruoyu_data/HKUSTGZ')
    
    for seq_type in ['train', 'val']:
        for seq_name in os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'image')):
                for frame_name in os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'image', seq_name)):
                    assert len(os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'image', seq_name, frame_name))) == len(os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'label_id', seq_name, frame_name)))
                    assert len(os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'image', seq_name, frame_name))) == len(os.listdir(os.path.join('/data/kin/ruoyu_data/HKUSTGZ', seq_type, 'label_color', seq_name, frame_name)))
                    
                    