import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import Config

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/..'


class VignetteCorrector():
    def __init__(self):
        # Configs
        self.cfg = Config()
        self.build_table = self.cfg.build_table
        self.image_width = self.cfg.image_width
        self.image_height = self.cfg.image_height
        self.center_x = self.cfg.center_x
        self.center_y = self.cfg.center_y
        self.params = np.array(self.cfg.params, dtype=np.float32)

        self.source_path = self.cfg.source_path
        self.result_path = self.cfg.result_path

        # Buffers
        self.table = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)

        # Get correction table
        if self.build_table:
            self.getCorrectionTable()
            self.saveLUT()
        else:
            self.loadLUT()
        
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

    def loadLUT(self):
        try:
            self.table = np.load(rsg_root+'/rsc/lookup_table/table.npy')
            print("Lookup table loaded")
        except:
            print("Failed to load lookup table")

    def saveLUT(self):
        try:
            np.save(rsg_root+'/rsc/lookup_table/table.npy', self.table)
            print("Lookup table saved")
        except:
            print("Failed to save lookup table")

    def getCorrectionTable(self):
        print("Generating lookup table")
        for xx in range(self.image_height):
            for yy in range(self.image_width):
                for ch in range(3):
                    dist = self.getPixelDistance(xx, yy, self.center_x, self.center_y)
                    value = \
                        self.params[ch, 0] + \
                        self.params[ch, 1] * (dist ** 2) + \
                        self.params[ch, 2] * (dist ** 4) + \
                        self.params[ch, 3] * (dist ** 6)
                    
                    self.table[xx, yy, ch] = self.params[ch, 0] / value

    def getPixelDistance(self, x, y, cx, cy):
        return np.sqrt( (x - cx) ** 2 + (y - cy) ** 2 )

    def rectifyLUT(self, image):
        image = image * self.table
        return np.clip(image, a_min=0, a_max=255).astype(np.uint8)
    
    def rectify_batch(self):
        images = glob.glob(self.source_path+'/*.png')

        for img_file in images:
            img_name = img_file.split('/')[-1]
            print("Processing image:", img_name)
            image = self.readImage(img_file)
            if image is None:
                continue

            image_rectified = self.rectifyLUT(image)
            self.saveImage(self.result_path+'/vig_'+img_name, image_rectified)


if __name__ == "__main__":
    corr = VignetteCorrector()

    corr.rectify_batch()