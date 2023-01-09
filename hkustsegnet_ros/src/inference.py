import os
import time
import glob
from pathlib import Path
import warnings
import argparse

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
import cv2

from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer

warnings.filterwarnings('ignore')

CONFIG_PATH = '/home/hkustgz_segnet/configs/cityscapes/H_48_D_4_prob_proto.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
