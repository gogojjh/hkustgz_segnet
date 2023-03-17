from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import numpy as np
import os

import sys
sys.path.append('/home/hkustgz_segnet/segnet')
from lib.utils.tools.logger import Logger as Log


class ImgWhiteBalancer(object):
    def __init__(self, args):
        super(ImgWhiteBalancer, self).__init__()
        self.args = args
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
    
    def __white_balance(self, img, mode=5):
        """
        白平衡处理(默认为1均值、2完美反射、3灰度世界、4基于图像分析的偏色检测及颜色校正、5动态阈值)
        """
        # 读取图像
        b, g, r = cv2.split(img)
        # 均值变为三通道
        h, w, c = img.shape
        if mode == 2:
            # 完美反射白平衡 ---- 依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
            output_img = img.copy()
            sum_ = np.double() + b + g + r
            hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
            Y = 765
            num, key = 0, 0
            ratio = 0.01
            while Y >= 0:
                num += hists[Y]
                if num > h * w * ratio / 100:
                    key = Y
                    break
                Y = Y - 1

            sumkey = np.where(sum_ >= key)
            sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
            times = len(sumkey[0])
            avg_b, avg_g, avg_r = sum_b / times, sum_g / times, sum_r / times

            maxvalue = float(np.max(output_img))
            output_img[:, :, 0] = output_img[:, :, 0] * maxvalue / int(avg_b)
            output_img[:, :, 1] = output_img[:, :, 1] * maxvalue / int(avg_g)
            output_img[:, :, 2] = output_img[:, :, 2] * maxvalue / int(avg_r)
        elif mode == 3:
            # 灰度世界假设
            b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
            # 需要调整的RGB分量的增益
            k = (b_avg + g_avg + r_avg) / 3
            kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
            ba, ga, ra = b * kb, g * kg, r * kr

            output_img = cv2.merge([ba, ga, ra])
        elif mode == 4:
            # 基于图像分析的偏色检测及颜色校正
            I_b_2, I_r_2 = np.double(b) ** 2, np.double(r) ** 2
            sum_I_b_2, sum_I_r_2 = np.sum(I_b_2), np.sum(I_r_2)
            sum_I_b, sum_I_g, sum_I_r = np.sum(b), np.sum(g), np.sum(r)
            max_I_b, max_I_g, max_I_r = np.max(b), np.max(g), np.max(r)
            max_I_b_2, max_I_r_2 = np.max(I_b_2), np.max(I_r_2)
            [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
            [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
            b0 = np.uint8(u_b * (np.double(b) ** 2) + v_b * b)
            r0 = np.uint8(u_r * (np.double(r) ** 2) + v_r * r)
            output_img = cv2.merge([b0, g, r0])
        elif mode == 5:
            # 动态阈值算法 ---- 白点检测和白点调整
            # 只是白点检测不是与完美反射算法相同的认为最亮的点为白点，而是通过另外的规则确定
            def con_num(x):
                if x > 0:
                    return 1
                if x < 0:
                    return -1
                if x == 0:
                    return 0

            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # YUV空间
            (y, u, v) = cv2.split(yuv_img)
            max_y = np.max(y.flatten())
            sum_u, sum_v = np.sum(u), np.sum(v)
            avl_u, avl_v = sum_u / (h * w), sum_v / (h * w)
            du, dv = np.sum(np.abs(u - avl_u)), np.sum(np.abs(v - avl_v))
            avl_du, avl_dv = du / (h * w), dv / (h * w)
            radio = 0.5  # 如果该值过大过小，色温向两极端发展

            valuekey = np.where((np.abs(u - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du)
                                | (np.abs(v - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv))
            num_y, yhistogram = np.zeros((h, w)), np.zeros(256)
            num_y[valuekey] = np.uint8(y[valuekey])
            yhistogram = np.bincount(np.uint8(num_y[valuekey].flatten()), minlength=256)
            ysum = len(valuekey[0])
            Y = 255
            num, key = 0, 0
            while Y >= 0:
                num += yhistogram[Y]
                if num > 0.1 * ysum:  # 取前10%的亮点为计算值，如果该值过大易过曝光，该值过小调整幅度小
                    key = Y
                    break
                Y = Y - 1

            sumkey = np.where(num_y > key)
            sum_b, sum_g, sum_r = np.sum(b[sumkey]), np.sum(g[sumkey]), np.sum(r[sumkey])
            num_rgb = len(sumkey[0])

            b0 = np.double(b) * int(max_y) / (sum_b / num_rgb)
            g0 = np.double(g) * int(max_y) / (sum_g / num_rgb)
            r0 = np.double(r) * int(max_y) / (sum_r / num_rgb)

            output_img = cv2.merge([b0, g0, r0])
        else:
            # 默认均值  ---- 简单的求均值白平衡法
            b_avg, g_avg, r_avg = cv2.mean(b)[0], cv2.mean(g)[0], cv2.mean(r)[0]
            # 求各个通道所占增益
            k = (b_avg + g_avg + r_avg) / 3
            kb, kg, kr = k / b_avg, k / g_avg, k / r_avg
            b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
            g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
            r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
            output_img = cv2.merge([b, g, r])
        output_img = np.uint8(np.clip(output_img, 0, 255))
        return output_img

    def __get_img_list(self, dir_name):
        filename_list = list()
        for item in sorted(os.listdir(dir_name)):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append(dir_name ,os.path.join('{}/{}'.format(item, filename)))
            else: 
                filename_list.append(os.path.join(dir_name ,item))
                
        return filename_list
    
    def process_img_batch(self):
        filename_list = self.__get_img_list(self.input_dir)
        Log.info('Number of input images: {}'.format(len(filename_list)))
        for img_file in filename_list:
            img_name, extension = os.path.splitext(img_file.split('/')[-1])
            img = cv2.imread(img_file)
            # print('start white balancing for {}'.format(img_file.split('/')[-1]))
            balanced_img = self.__white_balance(img)
            
            cv2.imwrite(os.path.join(self.output_dir, '{}.png'.format(img_name)), balanced_img)
            Log.info('Saving {}.png'.format(img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='/data/Data/kin/ruoyu_data/HKUSTGZ/frame01', type=str, dest='output_dir', help='directory of input image.')
    parser.add_argument('--input_dir', default='/data/Data/jjiao/dataset/FusionPortable_dataset_develop/sensor_data/mini_hercules/20230309_hkustgz_campus_road_day/data/frame_cam01/image/data', type=str,
                        dest='input_dir', help='directory of output image.')
    args = parser.parse_args()
    
    white_balancer = ImgWhiteBalancer(args)
    white_balancer.process_img_batch()