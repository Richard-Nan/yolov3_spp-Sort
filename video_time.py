import numpy as np
import cv2
import os
import time
import numpy as np
import os
import json
import time
import random
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs
import argparse
import time
import cv2
import os
import glob

def video_demo():
    # 加载已经训练好的模型路径，可以是绝对路径或者相对路径
    # 加载yolo训练的标签
    labelsPath = os.path.sep.join(["data/my_data_label.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # 初始化颜色列表以表示每个可能的类标签
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),
                               dtype="uint8")

    # 加载在COCO数据集上训练的YOLO对象检测器
    # 加载模型

    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "./weights/yolov3spp-79.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    # init
    # 读入待检测的图像
    # 0是代表摄像头编号，只有一个的话默认为0
    capture = cv2.VideoCapture(0)
    while (True):
        ref, image = capture.read()
        (H, W) = image.shape[:2]

        img = img_utils.letterbox(image, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]  # only get inference result
        t2 = torch_utils.time_synchronized()

        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], image.shape).round()

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        pil_img = Image.fromarray(image[:, :, ::-1])


        idxs = np.greater(scores, 0.1)
        boxes = bboxes[idxs]
        classes = classes[idxs]
        scores = scores[idxs]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # 提取边界框坐标
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                n = random.randint(0, 199)

                color = [int(c) for c in COLORS[n % len(COLORS)]]

                # [x,y,z]
                cv2.rectangle(image, (x, y), (w, h), color, 2)
                # 在原图上绘制边框和类别

                text = "{}: {:.4f}".format(LABELS[classes[i]], scores[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("Image", image)
                # 等待30ms显示图像，若过程中按“ESC”退出
                c = cv2.waitKey(30) & 0xff
                if c == 27:
                    capture.release()
                    break


video_demo()
