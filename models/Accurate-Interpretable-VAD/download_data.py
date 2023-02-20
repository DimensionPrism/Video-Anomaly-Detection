import os
import argparse
import subprocess

from utils.dataset_utils import count_frames, extract_shanghaitech_frames

def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_directory', type=str, default='./data', help='directory for downloaded data')
    argparser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    argparser.add_argument("--download_all", type=bool, default=False, help='download all datasets')

    args = argparser.parse_args()

    return args

def download_data(dataset):
    p = subprocess.run(["sh", f"./data/{dataset}.sh"])


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