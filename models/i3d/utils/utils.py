import os
import cv2
import glob
import argparse
import subprocess
import numpy as np
from pathlib import *


def extract_custom_dataset_frames(dataset_name, dataset_root):
    print(f"extracting {dataset_name} frames...")

    train_videos_path = f"{dataset_root}/{dataset_name}/training/videos/"
    films = list()
    files = (x for x in Path(train_videos_path).iterdir() if x.is_file())
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
            name = f"{dataset_root}/{dataset_name}/training/frames/{mapp}/{count}.jpg"
            if not os.path.isdir(f"{dataset_root}/{dataset_name}/training/frames/{mapp}"):
                os.makedirs(f"{dataset_root}/{dataset_name}/training/frames/{mapp}", exist_ok=True)
            cv2.imwrite(name, image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        print(f"Frames extracted from {mapp}!")
    
    test_videos_path = f"{dataset_root}/{dataset_name}/testing/videos/"
    films = list()
    files = (x for x in Path(test_videos_path).iterdir() if x.is_file())
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
            name = f"{dataset_root}/{dataset_name}/testing/frames/{mapp}/{count}.jpg"
            if not os.path.isdir(f"{dataset_root}/{dataset_name}/testing/frames/{mapp}"):
                os.makedirs(f"{dataset_root}/{dataset_name}/testing/frames/{mapp}", exist_ok=True)
            cv2.imwrite(name, image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        print(f"Frames extracted from {mapp}!")
    
    print(f"{dataset_name} frames extracted!")