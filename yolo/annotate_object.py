from __future__ import division

import argparse
import cv2
from darknet import Darknet
import numpy as np
import pandas as pd
import pickle as pkl
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from util import *

from preprocess import prep_image, inp_to_image


def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def draw_annotation(x, img, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    # color = random.choice(colors)
    color = colors[cls]
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


class VideoWriter:
    def __init__(self, path, fps, height, width):
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        self.out = cv2.VideoWriter(path, int(fourcc), fps, (int(width), int(height)))

    def write(self, img):
        self.out.write(img)
        return True

    def release(self):
        self.out.release()


class VideoViewer:
    def write(self, img):
        cv2.imshow("frame", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return False
        else:
            return True

    def release(self):
        pass


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3')
    parser.add_argument("--config", dest="config_file", required=True)
    parser.add_argument("--weights", dest="weights_file", required=True)
    parser.add_argument("--classes", dest="classes_file", required=True)
    parser.add_argument("--input", dest="input_file", required=True)
    parser.add_argument("--output", dest="output_file", required=True)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    # Configuration
    args = arg_parse()
    config_file = args.config_file
    weights_file = args.weights_file
    classes_file = args.classes_file
    input_file = args.input_file
    output_file = args.output_file
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    classes = load_classes(classes_file)
    num_classes = len(classes)

    # Model
    CUDA = torch.cuda.is_available()

    model = Darknet(config_file)
    model.load_weights(weights_file)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    # Input Source
    cap = cv2.VideoCapture(input_file)
    assert cap.isOpened(), 'Cannot capture source'
    fps    = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output
    writer = VideoWriter(output_file, fps, height, width)

    colors = pkl.load(open("pallete", "rb"))
    random.shuffle(colors)

    # Each Frame
    for index in tqdm(range(count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Get image
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1,2)

        # Detect object
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(output) != int:
            # Draw detected object to image
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            for x in output:
                draw_annotation(x, orig_im, colors)

        if not writer.write(orig_im):
            break

    # Finalize
    writer.release()
