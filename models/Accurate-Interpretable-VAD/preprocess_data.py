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
from utils.preprocess_utils.optical_flow_utils import extract_flows

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data', help='directory for downloaded data')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    
    parser.add_argument("--download_mode", type=int, default=1, help='0: do not download, 1: download specific data, 2: download all')

    parser.add_argument("--download_pretrained", type=bool, default=False, help='download pretrained model')

    # configs for object detection
    parser.add_argument("--object_detection", type=int, default=1, help='0: no object deteciton, 1: perfor object detection for specific data, 2: perform object detection for all data')

    # configs for optical flow
    parser.add_argument("--extract_flow", type=int, default=1, help='0: no optical flow, 1: optical flow for specific data, 2: optical flow for all data')

    # configs for feature extraction
    parser.add_argument("--feature_extraction", type=int, default=1, help='0: no feature extraction, 1: extract features from specific data, 2: extract features from all data')

    # configs for using custom dataset
    parser.add_argument("--custom_dataset", type=bool, default=False, help='use custom dataset')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = argparser()
    if not args.custom_dataset:
        # download data and frame preprocess
        dataset_names = ["ped2", "avenue", "shanghaitech"]
        if args.download_mode == 1:
            dataset_names = [args.dataset_name]
        else:
            dataset_names = []
        print("downloading data...")
        for dataset_name in dataset_names:
            download_data(dataset_name)
            if dataset_name == "shanghaitech":
                extract_shanghaitech_frames(args)
            count_frames(args, dataset_name)
        print("data downloaded!\n")
            
        # download pretrained model for object detection and optical flow
        if args.download_pretrained:
            download_pretrained()

        # object detection
        dataset_names = ["ped2", "avenue", "shanghaitech"]
        if args.object_detection == 1:
            dataset_names = [args.dataset_name]
        else:
            dataset_names = []
        for dataset_name in dataset_names:
            print(f"performing object detection for {dataset_name}...")
            extract_bboxes(dataset_name, args.data_root)
            print(f"{dataset_name} object detection completed!\n")

        # optical flow
        dataset_names = ["ped2", "avenue", "shanghaitech"]
        if args.extract_flow == 1:
            dataset_names = [args.dataset_name]
        else:
            dataset_names = []
        for dataset_name in dataset_names:
            print(f"extracting optical flows from {dataset_name}...")
            extract_flows(dataset_name, args.data_root)
            print(f"{dataset_name} optical flows extracted!\n")

        # feature extraction
        dataset_names = ["ped2", "avenue", "shanghaitech"]
        if args.feature_extraction == 1:
            dataset_names = [args.dataset_name]
        else:
            dataset_names = []
        for dataset_name in dataset_names:
            print(f"extracting features from {dataset_name}...")
            extract_features(dataset_name, args.data_root)
            print(f"{dataset_name} features extracted!\n")
    else:
        dataset_name = "custom_dataset"
        extract_custom_dataset_frames(dataset_name, args.data_root)
        extract_bboxes(dataset_name, args.data_root)
        extract_flows(dataset_name, args.data_root)
        extract_features(dataset_name, args.data_root)