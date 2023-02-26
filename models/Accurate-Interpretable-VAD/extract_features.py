import os
import cv2
import gdown
import torch
import argparse
import torchvision

import numpy as np

from model.CLIP import CLIP
from tqdm import tqdm
from model.feature_extraction import extract_velocity, extract

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str, default='./data', help='directory for downloaded data')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--download_bboxes", type=bool, default=True, help='download detected obj files')
    
    args = parser.parse_args()

    return args

def download_bboxes(args):
    if args.dataset_name == "ped2":
        train_bboxes_url= "https://drive.google.com/file/d/1Kjn3xrKq7CpCyLG0tKRsdqghxzUurv_I/view?usp=share_link"
        test_bboxes_url = "https://drive.google.com/file/d/1gf5unVRc1HjC1qEwzp7jW3Wo4GTROHQc/view?usp=share_link"
    train_bboxes_output = f"{args.data_directory}/{args.dataset_name}/{args.dataset_name}_bboxes_train.npy"
    test_bboxes_output = f"{args.data_directory}/{args.dataset_name}/{args.dataset_name}_bboxes_test.npy"
    if not os.path.exists(train_bboxes_output):
        gdown.download(train_bboxes_url, train_bboxes_output, quiet=True, fuzzy=True)
    if not os.path.exists(test_bboxes_output):
        gdown.download(test_bboxes_url, test_bboxes_output, quiet=True, fuzzy=True)
    


def run(args, root):
    extracted_features = extract(args, root)
    train_velocity = extracted_features[0]
    train_feature_space = extracted_features[1]
    test_velocity = extracted_features[2]
    test_feature_space = extracted_features[3]

    np.save(f'{args.data_directory}/extracted_features/{args.dataset_name}/train/velocity.npy', train_velocity)
    np.save(f'{args.data_directory}/extracted_features/{args.dataset_name}/train/deep_features.npy', train_feature_space)

    np.save(f'{args.data_directory}/extracted_features/{args.dataset_name}/test/velocity.npy', test_velocity)
    np.save(f'{args.data_directory}/extracted_features/{args.dataset_name}/test/deep_features.npy', test_feature_space)


if __name__ == "__main__":
    root = './data/'
    args = argparser()
    if args.download_bboxes:
        download_bboxes(args)
    run(args, root)