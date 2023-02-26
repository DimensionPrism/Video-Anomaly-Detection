import os
import cv2
import gdown
import torch
import argparse
import subprocess
import torchvision

import numpy as np

from model.CLIP import CLIP
from tqdm import tqdm
from utils.preprocess_utils.download_utils import *
from utils.preprocess_utils.object_detection_utils import *
from utils.preprocess_utils.feature_extraction_utils import extract_features
from utils.preprocess_utils.optical_flow_utils import install_flownet2, extract_flows

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default='./data', help='directory for downloaded data')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    
    parser.add_argument("--download_mode", type=int, default=1, help='0: do not download, 1: download specific data, 2: download all')
    
    parser.add_argument("--preprocess_mode", type=int, default=1, help='do not preprocess, 1: preprocess specific data, 2: preprocess all')

    # configs for object detection
    parser.add_argument("--install_detectron2", type=bool, default=False, help='install detectron2')
    parser.add_argument("--download_pretrained", type=bool, default=True, help='download pretrained model')

    # configs for optical flow
    parser.add_argument("--install_flownet2", type=bool, default=False, help='install flownet2')

    # configs for using custom dataset
    parser.add_argument("--custom_dataset", type=bool, default=False, help='use custom dataset')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = argparser()

    # download data and frame preprocess
    if args.download_mode == 2:
        dataset_names = ["ped2", "avenue", "shanghaitech"]
    elif args.download_mode == 1:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = []
    
    for dataset_name in dataset_names:
        download_data(dataset_name)
        if dataset_name == "shanghaitech":
            extract_shanghaitech_frames(args)
        count_frames(args, dataset_name)
        
    # download pretrained model for object detection and optical flow
    if args.download_pretrained:
        download_pretrained()

    # object detection
    if args.preprocess_mode == 2:
        dataset_names = ["ped2", "avenue", "shanghaitech"]
    elif args.preprocess_mode == 1:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = []

    if args.install_detectron2:
        install_detectron2()

    for dataset_name in dataset_names:
        extract_bboxes(dataset_name, args.data_directory)

    # optical flow
    if args.install_flownet2:
        install_flownet2()
    
    for dataset_name in dataset_names:
        extract_flows(dataset_name, args.data_directory)

    # feature extraction
    for dataset_name in dataset_names:
        extract_features(args, args.data_directory)