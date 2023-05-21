import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from lib.utils.tools.logger import Logger as Log
from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.datasets.tools.collate import collate
import lib.datasets.tools.transforms as trans
import sys
sys.path.append('/home/catkin_ws/src/segnet')
# from custom_imagemsg.msg import CustomImage


class ROSProcessor():
    def __init__(self, configer, tester):
        self.configer = configer
        self.img_topic = self.configer.get('ros', 'image_topic')
        self.sem_img_topic = self.configer.get('ros', 'sem_image_topic')
        self.sem_rgb_img_topic = self.configer.get('ros', 'sem_rgb_image_topic')
        self.uncer_img_topic = self.configer.get('ros', 'uncer_image_topic')
        self.msg_type = self.configer.get('ros', 'msg_type')

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])
        self.trans_dict = self.configer.get('test', 'data_transformer')

        self.img_num = 0
        size_mode = self.configer.get('test', 'data_transformer')['size_mode']
        self.is_stack = (size_mode != 'diverse_size')

        self.bridge = CvBridge()
        if self.configer.get('ros', 'msg_type') == 'sensor_msgs/CompressedImage':
            self.compressed_image = True
        else:
            self.compressed_image = False

        self.model = tester

    def init_ros(self):
        if self.compressed_image:
            self.img_sub = rospy.Subscriber(
                self.img_topic, CompressedImage, self._image_callback, queue_size=1,
                buff_size=52428800)
        else:
            self.img_sub = rospy.Subscriber(
                self.img_topic, Image, self._image_callback, queue_size=1, buff_size=52428800)
        self.sem_img_pub = rospy.Publisher(self.sem_img_topic, Image, queue_size=1)
        self.sem_rgb_img_pub = rospy.Publisher(self.sem_rgb_img_topic, Image, queue_size=1)
        self.uncer_img_pub = rospy.Publisher(self.uncer_img_topic, Image, queue_size=1)

    def pub_semimg_msg(self, sem_img, ori_header):
        import numpy as np
        bridge = CvBridge()
        try:
            sem_img_16UC1 = sem_img.astype(np.uint16)
            self.sem_img_pub.publish(bridge.cv2_to_imgmsg(
                sem_img_16UC1, 'mono16', header=ori_header))
            Log.info('pub sem img topic')
        except CvBridgeError as e:
            Log.error(e)

    def pub_semrgbimg_msg(self, sem_rgb_img, ori_header):
        import numpy as np
        bridge = CvBridge()
        try:
            self.sem_rgb_img_pub.publish(bridge.cv2_to_imgmsg(
                sem_rgb_img, 'bgr8', header=ori_header))
            Log.info('pub sem rgb img topic')
        except CvBridgeError as e:
            Log.error(e)

    def pub_uncerimg_msg(self, uncer_img, ori_header):
        bridge = CvBridge()
        try:
            self.uncer_img_pub.publish(bridge.cv2_to_imgmsg(uncer_img, encoding="bgr8",
                                                            header=ori_header))
            Log.info('pub uncertainty img topic')
        except CvBridgeError as e:
            Log.error(e)

    def _test(self, data_dict, ori_header):
        self.model.get_ros_batch_data(data_dict)  # seg test dataloader as the data_dict from ros

        sem_img_ros, uncer_img_ros_list = self.model.test()  # test phase of the model
        assert uncer_img_ros_list[0].shape[-1] == 3
        assert len(sem_img_ros) == 2

        self.pub_uncerimg_msg(uncer_img_ros_list[0], ori_header)
        self.pub_semimg_msg(sem_img_ros[0], ori_header)
        self.pub_semrgbimg_msg(sem_img_ros[1], ori_header)

    def _prepare_data_dict(self, img, timestamp):
        ''' 
        To imitate the pytorch dataloader.
        Get an image(batch_size=1) from ros callback function,
        then transform the image and form the data dict as in the 
        original dataloader.
        '''
        img_size = ImageHelper.get_size(img)

        # transform img from ros msg
        if self.img_transform is not None:
            img = self.img_transform(img)
        meta = dict(
            ori_img_size=img_size,
            border_size=img_size,
        )

        data_dict = dict(
            img=DataContainer(img, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(
                str(timestamp), stack=False, cpu_only=True),
        )

        # collate_fn in dataloader to make img to fixed size

        data_dict = [data_dict]
        data_dict = collate(data_dict, trans_dict=self.trans_dict)

        self.img_num += 1

        return [data_dict]

    def _image_callback(self, msg, mode='rgb'):
        ''' 
        When new rosmsg comes:
        - Convert to img/transform/form data_loader-like data_dict
        - Send data_dict to model for inference
        '''
        try:
            if not self.compressed_image:
                if self.configer.get('data', 'input_mode') == 'BGR':
                    img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                    if mode == 'rgb':
                        img = ImageHelper.bgr2rgb(img)
                if self.configer.get('data', 'input_mode') == 'RGB':
                    img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            else:
                if self.configer.get('data', 'input_mode') == 'BGR':
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                    if mode == 'rgb':
                        img = ImageHelper.bgr2rgb(img)
                if self.configer.get('data', 'input_mode') == 'RGB':
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            Log.error(e)

        timestamp = msg.header.stamp
        ori_header = msg.header

        data_dict = self._prepare_data_dict(img, timestamp)

        self._test(data_dict, ori_header)
