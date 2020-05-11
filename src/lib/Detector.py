from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import time

"""
opt.fix_res
opt.pad
opt.reg_offset
opt.flip_test
"""
from lib.models import load_model, save_model, create_model
from utils import image_process as ip
from utils import decode
# from external.nms import soft_nms

class Detector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print("createing model..........")
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 20
        self.num_classes = opt.num_classes
        self.opt = opt
        self.scales = opt.test_scales

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height*scale)
        new_width = int(width*scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)  # center
            s = max(height, width) * 1.0  # scale
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)


        ## 做仿射变换
        trans_input = ip.get_affine_transform(c, s, 0, [inp_width ,inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image/255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c,
                's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio
                }
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]  # what is output
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = decode.ctdet_decode(hm, wh, reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = decode.ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], self.opt.num_classes
        )
        for i in range(1, self.opt.num_classes + 1):
            dets[0][i] = np.array(dets[0][i], dtype=np.float32).reshape(-1, 5)
            dets[0][i][:, :4] /= scale
        return dets[0]

    def merge_output(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)

            # soft_nms(results[j], Nt=0.5, method=2)

            det_new = results[j][:, :4].tolist()
            score_new = results[j][:, 4].tolist()
            if len(det_new) != 0:
                det_new = torch.tensor(det_new, dtype=torch.float)
                # print(det_new)
                score_new = torch.tensor(score_new, dtype=torch.float)
                keep = soft_nms_pytorch(det_new, score_new)
                keep = keep.detach().cpu().numpy().tolist()
                results[j] = results[j][keep]

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)]
        )
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run(self, image_path_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        start_time = time.time()
        pre_processed = False # ?
        if isinstance(image_path_tensor, np.ndarray):
            image = image_path_tensor
        elif type(image_path_tensor) == type(''):
            image = cv2.imread(image_path_tensor)
        else:
            image = image_path_tensor['image'][0].numpy()
            pre_processed_images = image_path_tensor
            pre_processed = True
        load_time += (time.time() - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_output(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}

def soft_nms_pytorch(dets, box_scores, sigma=0.1, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [x1,y1,x2,y2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep
