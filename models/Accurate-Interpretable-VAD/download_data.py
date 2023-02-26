import os
import argparse
import subprocess

from utils.preprocess_utils import count_frames, extract_shanghaitech_frames

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', type=str, default='./data', help='directory for downloaded data')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--download_all", type=bool, default=False, help='download all datasets')

    args = parser.parse_args()

    return args

def download_data(dataset):
    p = subprocess.run(["sh", f"./utils/download_utils/{dataset}.sh"])


if __name__ == "__main__":
    args = argparser()
    if not args.download_all:
        datasets = [args.dataset_name]
    else:
        datasets = ["ped2", "avenue", "shanghaitech"]
    for dataset in datasets:
        download_data(dataset)
        if dataset == "shanghaitech":
            print(f"Extracting {dataset} frames...")
            extract_shanghaitech_frames(args)
            print(f"Frames extracted!")
        print(f"Counting {dataset} frames...")
        count_frames(args, dataset)
        print(f"Frames counted!")