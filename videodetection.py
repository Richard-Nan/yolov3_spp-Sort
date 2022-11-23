import numpy as np
import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs
import argparse
import imutils
import time
import cv2
import os
import glob
from sort import *
#
files = glob.glob('output/*.png')
for f in files:
    os.remove(f)



tracker = Sort()
memory = {}
# line = [(465,595), (1447,595)]    # video_work 分割线
line = [(0,790),(1920,790)]	# video_test1_1920x1080分割线   # video_test1_1920x1080分割线
# line = [(100,1700),(3100,1700)]	# video_test2_4K 分割线
counter = 0


# 检测线段 AB 和 CD 是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


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
weights = "./weights/yolov3spp-29.pt"  # 改成自己训练好的权重文件
json_path = "./data/pascal_voc_classes.json"  # json标签文件
img_path = "test.jpg"
assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

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
with torch.no_grad():
    img = torch.zeros((1, 3, img_size, img_size), device=device)
    model(img)
    # 初始化视频流、指向输出视频文件的指针和帧
    vs = cv2.VideoCapture("./input/highway.mp4")
    writer = None
    (W, H) = (None, None)
    frameIndex = 0
    # 计算视频中的总帧数
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # 帧出错时
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    # 循环视频流中的帧
    while True:
        # 读取下一帧
        (grabbed, frame) = vs.read()

        # 检测是否到达最后一帧，如果下一帧没有锚框，则为最后一帧
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        img = img_utils.letterbox(frame, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
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
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4],frame.shape).round()

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        pil_img = Image.fromarray(frame[:, :, ::-1])

        idxs = np.greater(scores, 0.1)
        boxes = bboxes[idxs]
        classes = classes[idxs]
        scores = scores[idxs]

        dets = bboxes
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        if np.size(dets) == 0:
            continue
        else:
            tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # 提取边界框坐标
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                # [x,y,z]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    if intersect(p0, p1, line[0], line[1]):
                        counter += 1

                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                i += 1

        # 绘制分割线
        cv2.line(frame, line[0], line[1], (0, 255, 0), 2)

        # 绘制计数显示
        cv2.putText(frame, str(counter), (430, 205), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.0, (64, 244, 205), 5)
        cv2.putText(frame, "counter:", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (34, 139, 34), 5)

        # 存储绘制的每一帧
        cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

        # 检测视频是否写入
        if writer is None:
            # 初始化视频写入
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("output/output.mp4", fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

            # if total > 0:
            #     elap = (end - start)
            #     print("[INFO] single frame took {:.4f} seconds".format(elap))
            #     print("[INFO] estimated total time to finish: {:.4f}".format(
            #         elap * total))

        # 将每帧写入硬盘
        writer.write(frame)

        frameIndex += 1

        if frameIndex >= 40000:
            print("[INFO] cleaning up...")
            writer.release()
            vs.release()
            exit()

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()
    cv2.destroyAllWindows()