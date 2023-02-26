import os
import cv2
import glob
import argparse
import subprocess
import numpy as np
from pathlib import *


def download_data(dataset):
    p = subprocess.run(["sh", f"./utils/preprocess_utils/{dataset}.sh"])

def count_frames(args, dataset_name):
    print(f"counting {dataset_name} frames...")

    root = f'{args.data_root}/{dataset_name}/training/frames/'

    train_clip_lengths = []
    test_clip_lengths = []

    folders = glob.glob(os.path.join(root, '*'))
    folders.sort()
    lengths = []
    count = 0
    for i, folder in enumerate(folders):
        video_folder = folder.split('/')[-1]
        video_frames = glob.glob(os.path.join(root, video_folder, '*.jpg'))
        video_frames.sort()
        count += len(video_frames)
        train_clip_lengths.append(count)

    root = f'{args.data_root}/{dataset_name}/testing/frames/'

    folders = glob.glob(os.path.join(root, '*'))
    folders.sort()
    lengths = []
    count = 0
    for i, folder in enumerate(folders):
        video_folder = folder.split('/')[-1]
        video_frames = glob.glob(os.path.join(root, video_folder, '*.jpg'))
        video_frames.sort()
        count += len(video_frames)
        test_clip_lengths.append(count)
    
    np.save(f'{args.data_root}/{dataset_name}/train_clip_lengths.npy', train_clip_lengths)
    np.save(f'{args.data_root}/{dataset_name}/test_clip_lengths.npy', test_clip_lengths)

    print(f"frames counted!")

def extract_shanghaitech_frames(args):
    print(f"extracting shanghaitech frames...")

    path_videos = f"{args.data_root}/shanghaitech/training/videos/"
    path_frames = f"{args.data_root}/shanghaitech/training/frames/"


    films = list()
    files = (x for x in Path(path_videos).iterdir() if x.is_file())
    for file in files:
        print(str(file.name).split(".")[0], "is a file!")
        films.append(file)

    for i, film in enumerate(films):
        count = 0
        vidcap = cv2.VideoCapture(str(film))
        success, image = vidcap.read()
        mapp = str(film.name).split(".")[0]
        print(f"Extracting frames from {mapp}...")
        while success:
            name = f"{args.data_root}/shanghaitech/training/frames/{mapp}/{count}.jpg"
            if not os.path.isdir(f"{args.data_root}/shanghaitech/training/frames/{mapp}"):
                os.makedirs(f"{args.data_root}/shanghaitech/training/frames/{mapp}", exist_ok=True)
            cv2.imwrite(name, image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        print(f"Frames extracted from {mapp}!")
    
    print("shanghaitech frames extracted!")