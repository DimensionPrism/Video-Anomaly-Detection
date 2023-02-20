import os
import cv2
import torch
import argparse
import torchvision

import numpy as np

from models.models import CLIP
from tqdm import tqdm
from models.feature_extraction import extract_velocity, extract

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    
    args = parser.parse_args()

    return args


def run(args, root):
    extracted_features = extract(args, root)
    train_velocity = extracted_features[0]
    train_feature_space = extracted_features[1]
    test_velocity = extracted_features[2]
    test_feature_space = extracted_features[3]

    np.save('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), train_velocity)
    np.save('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), train_feature_space)

    np.save('extracted_features/{}/test/velocity.npy'.format(args.dataset_name), test_velocity)
    np.save('extracted_features/{}/test/deep_features.npy'.format(args.dataset_name), test_feature_space)


if __name__ == "__main__":
    root = './data/'
    args = argparser()
    run(args, root)