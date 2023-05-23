import os
import glob
import cv2
import numpy as np
from image_similarity_measures.quality_metrics import fsim, rmse
import rosbag

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/..'


class ROSConfig():
    bag_file = 'gym01_2023-04-04-15-48-49.bag'
    image_topic = '/cam_lookdn/color/image_raw/compressed'
    

class Config():
    ros = ROSConfig()

    source_path = "/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence00/data/frame_cam00/image/data"
    result_path = "/data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230403_hkustgz_campus_road_day_sequence00/data/frame_cam00/image_annotated/data"

    decimation = 45
    sim_rescale = 0.05
    fsim_threshold = 0.375
    rmse_threshold = 0.012
    buffer_size = 3



class ImageSelector():
    def __init__(self):
        # Configs
        self.cfg = Config()

        self.source_path = self.cfg.source_path
        self.result_path = self.cfg.result_path

        self.decimation = self.cfg.decimation
        self.sim_rescale = self.cfg.sim_rescale
        self.fsim_threshold = self.cfg.fsim_threshold
        self.rmse_threshold = self.cfg.rmse_threshold
        self.buffer_size = self.cfg.buffer_size

        # self.bag_file = rosbag.Bag(self.source_path+self.cfg.ros.bag_file)
        
        # Buffers
        self.msg_counter = 0
        self.img_counter = 0
        self.image_buf = []
        
    def processBag(self):
        for topic, msg, t in self.bag_file.read_messages(topics=[self.cfg.ros.image_topic]):
            img = self.readImgMsg(msg)

            if self.msg_counter % self.decimation != 0:
                self.msg_counter += 1
                continue

            # Resize image for similarity check
            width = int(img.shape[1] * self.sim_rescale)
            height = int(img.shape[0] * self.sim_rescale)
            dim = (width, height)
            img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

            if self.msg_counter == 0:
                # Save the first image
                self.saveImage(self.result_path+str(self.img_counter)+'.png', img)
                self.img_counter += 1
                self.image_buf.append(img_resized)
            else:
                sim_fsim = -1e3
                sim_rmse = 1e3
                for img_buf in self.image_buf:
                    # Use FSIM and RMSE algorithm for image similarity measurement
                    # Paper: https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf
                    sim_fsim = max(sim_fsim, fsim(img_resized, img_buf))
                    sim_rmse = min(sim_rmse, rmse(img_resized, img_buf))
                
                if sim_fsim < self.fsim_threshold and sim_rmse > self.rmse_threshold:
                    self.saveImage(self.result_path+str(self.img_counter)+'.png', img)
                    self.img_counter += 1
                    self.image_buf.append(img_resized)
                    if len(self.image_buf) > self.buffer_size:
                        self.image_buf.pop(0)

            print("Progress:", self.msg_counter, end='\r')
            self.msg_counter += 1

        self.bag_file.close()

    def readImgMsg(self, msg):
        img_np = np.frombuffer(msg.data, np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        return img_cv

    def readImage(self, file_path):
        try:
            image = cv2.imread(file_path)
            return image
        except:
            print("Failed to read image:", file_path)
            return None
        
    def saveImage(self, file_path, image):
        try:
            cv2.imwrite(file_path, image)
            print("Image saved:", file_path)
        except:
            print("Failed to save image:", file_path)


if __name__ == "__main__":
    ims = ImageSelector()

    # ims.processBag()