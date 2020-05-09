from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import cv2
from lib.opts import opts
from lib.Detector import Detector
from lib.dataset.Pascal import PascalVOC

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

# sys.path.append("C:/User/13778/workshop/gitrepos/Air_Apron/src/")
sys.path.append("/gs/home/tongchao/zc/Air_Apron/src/")


def demo(opt):
    cls_map_id = ['__background__', 'plane', 'head', 'wheel', 'wings', 'stair',
                           'oil_car', 'person', 'cone', 'engine', 'traction', 'bus', 'queue', 'cargo']
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = PascalVOC
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    detector = Detector(opt)
    rets = []
    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            rets.append(ret['results'])
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            rets.append(ret['results'])
    dets_total = []
    for i in range(len(rets)):
        all_bboxes = rets[i]
        detections = []
        for cls_ind in all_bboxes:
            category_id = cls_map_id[cls_ind]
            for bbox in all_bboxes[cls_ind]:
                score = bbox[4]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                bbox_out = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]

                detection = {
                    "class": category_id,
                    "bbox": bbox_out,
                    "center": [(x2+x1)/2, (y1+y2)/2],
                    "size": [x2-x1, y2-y1],
                    "score": float("{:.2f}".format(score))
                }
                if score >= 0.02 and x1 > 0 and y2 < 576:
                    detections.append(detection)
        dets_total.append(detections)
    print(dets_total)
    return dets_total


def detection_demo(detector, img):
    ret = detector.run(img)
    time_str = ''
    for stat in time_stats:
        time_str += '{} {:.3f}s |'.format(stat, ret[stat])
    return ret, time_str


if __name__ == "__main__":
    opt = opts().parse()
    rets = demo(opt)
