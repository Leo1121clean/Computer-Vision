import rospy
import rosbag
import cv2
from cv_bridge import CvBridge
import pyrealsense2
from ctypes import *
import numpy as np
from PIL import Image

from tools.demo import *
import argparse
import os
import signal
import time
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from loguru import logger

sem = 0
def handler(signum, stack):
    global sem
    sem = 1
    print("ok")
    
# path='/home/frank/catkin_ws/src/cv_final/img/'

# bag_file = '/home/frank/catkin_ws/demo_video.bag'
# bag = rosbag.Bag(bag_file, "r")
# bag_data = bag.read_messages()

# bridge = CvBridge()
# for topic, msg, t in bag_data:
#     print(t)
#     if topic == "/camera/color/image_raw":
#         cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
#         timestr = "%.3f" %  msg.header.stamp.to_sec()
#         image_name = timestr + "_rgb" + ".jpg"
#         cv2.imwrite(path + image_name, cv_image)
    
#     if topic == "/camera/aligned_depth_to_color/image_raw":
#         cv_image = bridge.imgmsg_to_cv2(msg, "16UC1")
#         timestr = "%.3f" %  msg.header.stamp.to_sec()
#         image_name = timestr + "_depth" + ".jpg"
#         cv2.imwrite(path + image_name, cv_image)
#######################################################

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./haha.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/default/yolox_s.py",
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="yolox_s.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.80, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def bbox(image):
    args=make_parser().parse_args()
    exp=get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,)
    output, img_info = predictor.inference(image)
    if(output[0] == None):
        bbox = None
    else:
        imgwithbbox, bbox = predictor.visual(output[0], img_info)

    # print(imgwithbbox.shape)
    # cv2.imshow("img", imgwithbbox)
    # cv2.waitKey(0)
    return bbox


#######################################################
class Nodo(object):

    def __init__(self):
        self.path = '/home/frank/catkin_ws/src/cv_final/img/'
        self.bag_data = None
        self.time_record = [] # 紀錄時間戳

        self.depth_image = None
        self.rgb_image = None
        self.depth_br = CvBridge()
        self.rgb_br = CvBridge()
        self.computImageRGB = []
        self.computImageDP = []
        self.ImageDP = []

        ### S,V,T declaration ###
        self.distance = 0
        self.time_difference = 0
        self.velovity = 0

        ### intrinsic parameter ###
        self._intrinsics = pyrealsense2.intrinsics()
        self._intrinsics.width = 640
        self._intrinsics.height = 480
        self._intrinsics.ppx = 322.333831
        self._intrinsics.ppy = 238.768722
        self._intrinsics.fx = 617.044128
        self._intrinsics.fy = 617.0698852
        self._intrinsics.model  = pyrealsense2.distortion.none
        self._intrinsics.coeffs = [0,0,0,0,0]

    # 讀bag
    def bag_read(self):
        bag_file = '/home/frank/catkin_ws/demo_video.bag'
        bag = rosbag.Bag(bag_file, "r")
        self.bag_data = bag.read_messages()
    
    # 儲存每一幀的照片
    def image_save(self):
        timeRecord = []
        for topic, msg, t in self.bag_data:
            ### rgb image ###
            if topic == "/camera/color/image_raw":
                self.rgb_image = self.rgb_br.imgmsg_to_cv2(msg)
                timestr = "%.3f" %  msg.header.stamp.to_sec()
                image_name = timestr + "_rgb" + ".jpg"
                # image = Image.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
                # res_image = image.convert('P')
                # res_image.save(self.path + image_name)
                cv2.imwrite(self.path + image_name, self.rgb_image)
    
            ### depth image ###
            if topic == "/camera/aligned_depth_to_color/image_raw":
                self.depth_image = self.depth_br.imgmsg_to_cv2(msg)
                timestr = "%.3f" %  msg.header.stamp.to_sec()
                image_name = timestr + "_depth" + ".jpg"
                # image = Image.fromarray(cv2.cvtColor(self.depth_image, cv2.COLOR_BGR2RGB))
                # res_image = image.convert('P')
                # res_image.save(self.path + image_name)
                cv2.imwrite(self.path + image_name, self.depth_image)
            
            # 紀錄每一幀的時間
            timeRecord.append(timestr)
        
        for i in range(int(len(timeRecord)/2)):
            rgb_img = cv2.imread(self.path + str(timeRecord[i]) + "_rgb.jpg")
            dp_img = cv2.imread(self.path + str(timeRecord[i]) + "_depth.jpg")
            rgb_img_new = rgb_img.copy()
            check = bbox(rgb_img_new)
            if(check != None):
                self.computImageRGB.append(rgb_img)
                self.time_record.append(timeRecord[i])
                self.computImageDP.append(dp_img)
            if(len(self.computImageRGB) == 3):
                break

        for i in range(len(timeRecord) - 1, int(len(timeRecord)/2), -1):
            rgb_img = cv2.imread(self.path + str(timeRecord[i]) + "_rgb.jpg")
            dp_img = cv2.imread(self.path + str(timeRecord[i]) + "_depth.jpg")
            rgb_img_new = rgb_img.copy()
            check = bbox(rgb_img_new)
            if(check != None):
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", check)
                self.computImageRGB.append(rgb_img)
                # cv2.imshow("img", rgb_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                self.time_record.append(timeRecord[i])
                self.computImageDP.append(dp_img)
            if(len(self.computImageRGB) == 6):
                break
        
        if(len(self.computImageRGB) < 6):
            return 0
        else:
            return 1

    
    def convert_depth_to_phys_coord_using_realsense(self, human_center, human_depth):
        result = pyrealsense2.rs2_deproject_pixel_to_point(self._intrinsics, human_center, human_depth)
        xyz_points = []
        # xyz_points.append([-result[0]/1000,-result[1]/1000, result[2]/1000])
        xyz_points.append([-result[0],-result[1], result[2]])
        return xyz_points
    
    def calculation(self):
        # 前3幀和後3幀影像的時間 算速度用
        begin_image_time = [float(self.time_record[0]), float(self.time_record[1]), float(self.time_record[2])]
        end_image_time = [float(self.time_record[-3]), float(self.time_record[-2]), float(self.time_record[-1])]
        # 紀錄3D座標的list
        begin_point = []
        end_point = []
        # 前3幀和後3幀影像的人員3D座標 算速度用
        begin_avg = [0, 0, 0]
        end_avg = [0, 0, 0]
        
        # 計算前3幀人的3D座標
        for i in range(3):
            # rgb_img = cv2.imread(self.path + str(i) + "_rgb.jpg")
            # depth_img = cv2.imread(self.path + str(i) + "_depth.jpg") #會改成用rgb_img估測深度
            # human_center = [240, 320] #會改成用rgb_img做yolo
            human_info = bbox(self.computImageRGB[i])
            if(human_info != None):
                human_center = [int((human_info[0][0] + human_info[0][2])/2), int((human_info[0][1]+ human_info[0][3])/2)]
            else:
                return 0
            print("########################## human center: ", human_center)
            
            # human_depth = depth_img[240, 320, 0] 
            human_depth = self.ImageDP[i][human_center[1]][human_center[0]]
            
            points = self.convert_depth_to_phys_coord_using_realsense(human_center, human_depth)
            print("########################## begin position is:[{}, {}, {}]".format(points[0][0], points[0][1], points[0][2]))
            begin_point.append([points[0][0], points[0][1], points[0][2]])
        
        # 取前3幀人的3D座標平均
        for i in range(len(begin_image_time)):
            begin_avg[0] += begin_point[i][0]
            begin_avg[1] += begin_point[i][1]
            begin_avg[2] += begin_point[i][2]
        begin_avg = np.asarray(begin_avg)/3
        print("########################## begin average: ", begin_avg)
        
        # 計算後3幀人的3D座標
        for i in range(3,6):
            # rgb_img = cv2.imread(self.path + str(i) + "_rgb.jpg")
            # depth_img = cv2.imread(self.path + str(i) + "_depth.jpg") #會改成用rgb_img估測深度
            # human_center = [240, 320] #會改成用rgb_img做yolo
            human_info = bbox(self.computImageRGB[i])
            if(human_info != None):
                human_center = [int((human_info[0][0] + human_info[0][2])/2), int((human_info[0][1]+ human_info[0][3])/2)]
            else:
                return 0
            print("########################## human center: ", human_center)

            # human_depth = depth_img[240, 320, 0]
            human_depth = self.ImageDP[i][human_center[1]][human_center[0]]
            
            points = self.convert_depth_to_phys_coord_using_realsense(human_center, human_depth)
            print("########################## end position is:[{}, {}, {}]".format(points[0][0], points[0][1], points[0][2]))
            end_point.append([points[0][0], points[0][1], points[0][2]])
        
        # 取後3幀人的3D座標平均
        for i in range(len(end_image_time)):
            end_avg[0] += end_point[i][0]
            end_avg[1] += end_point[i][1]
            end_avg[2] += end_point[i][2]
        end_avg = np.asarray(end_avg)/3
        print("########################## end average: ", end_avg)

        # 計算速度
        begin_mean_time = np.mean(np.asarray(begin_image_time))
        end_mean_time = np.mean(np.asarray(end_image_time))
        print("")
        print("Begin mean time: ", begin_mean_time, "s")
        print("End mean time: ", end_mean_time, "s")
        self.distance = np.linalg.norm(end_avg - begin_avg)
        self.time_difference = end_mean_time - begin_mean_time
        self.velocity = self.distance / self.time_difference
        return 1
            
    def calculate_depth(self):
        global sem 
        
        for i in range(6):
            cv2.imwrite("/home/frank/Lite-Mono/test_img/" + "rgb_image{}.jpg".format(i), self.computImageRGB[i])
        while(1):
            if(sem == 1):
                break

        for i in range(6):
            self.ImageDP.append(np.load("/home/frank/Lite-Mono/test_img/" + "rgb_image{}_depth.npy".format(i)))

    def start(self):
        # img = cv2.imread("111.jpg")
        # output = bbox(img)
        # # print(int(output[0][0]))
        self.bag_read()
        flag = self.image_save()
        self.calculate_depth()
        if(flag != 1):
            return print("No person1111111111")
        flag = self.calculation()
        if(flag == 1):
            print("Distance: ", self.distance, "m")
            print("Elapsed time: ", self.time_difference, "s")
            print("Final velocity: ", self.velocity, "m/s")
        else:
            print("No person")


if __name__ == '__main__':
    print("PID is ", os.getpid())
    signal.signal(signal.SIGUSR1, handler)
    rospy.init_node("cv_final", anonymous=True)
    my_node = Nodo()
    my_node.start()