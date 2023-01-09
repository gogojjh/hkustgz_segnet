import lib.datasets.tools.transforms as trans
from lib.datasets.tools.collate import collate
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import rospy


class ROSProcessor():
    def __init__(self, configer, tester):
        self.configer = configer
        self.img_topic = self.configer.get('ros','image_topic')
        self.pub_img_topic = self.configer.get('ros', 'sem_image_topic')
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
        
    def pub_semimg_msg(self, sem_img):
        bridge = CvBridge()

        try:
            self.sem_img_pub.publish(bridge.cv2_to_imgmsg(sem_img, 'bgr8'))
            # self.sem_img_pub.publish(bridge.cv2_to_compressed_imgmsg(sem_img, 'bgr8'))
            Log.info('pub sem img topic')
        except CvBridgeError as e:
            Log.error(e)

    def init_ros(self):
        if self.compressed_image:
            self.img_sub = rospy.Subscriber(
            self.img_topic, CompressedImage, self._image_callback, queue_size=1, buff_size=2*24)
        else: 
            self.img_sub = rospy.Subscriber(
            self.img_topic, Image, self._image_callback, queue_size=1, buff_size=2*24)
        self.sem_img_pub = rospy.Publisher(self.pub_img_topic, Image, queue_size=20)

    def _test(self, data_dict):
        self.model.get_ros_batch_data(data_dict)  # seg test dataloader as the data_dict from ros

        sem_img_ros = self.model.test()  # test phase of the model
        
        assert len(sem_img_ros) == 1
        
        self.pub_semimg_msg(sem_img_ros[0])

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
                        cv_image = ImageHelper.bgr2rgb(img)
                if self.configer.get('data', 'input_mode') == 'RGB':
                    img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            else:
                if self.configer.get('data', 'input_mode') == 'BGR':
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                    if mode == 'rgb':
                        cv_image = ImageHelper.bgr2rgb(img)
                if self.configer.get('data', 'input_mode') == 'RGB':
                    img = self.bridge.compressed_imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            Log.error(e)

        timestamp = msg.header.stamp

        data_dict = self._prepare_data_dict(img, timestamp)

        self._test(data_dict)
