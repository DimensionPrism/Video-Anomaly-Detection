Skip to content
Search or jump to…
Pull requests
Issues
Codespaces
Marketplace
Explore
 
@Erikellerx 
DimensionPrism
/
CSCI-4962-Projects-in-ML-and-AI
Public
Fork your own copy of DimensionPrism/CSCI-4962-Projects-in-ML-and-AI
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Beta Try the new code view
CSCI-4962-Projects-in-ML-and-AI/Final_Project/src/utils/preprocess_utils.py /
@DimensionPrism
DimensionPrism add argparser to all training script
Latest commit c03f398 on Dec 8, 2022
 History
 1 contributor
182 lines (160 sloc)  6.76 KB
 

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from os import listdir
from shutil import copy, move
from os.path import join, exists
from src.utils.os_utils import remove_tree



def preprocess_widerface(raw_root, processed_root, pixel_threshold=0.75, process_test=True, remove_processed=False, keep_dir=False):
    print("Processing Wider Face dataset.")
    if not exists(raw_root):
        raise Exception("Wider Face dataset not found. Please check again!")
    dataset_types = ['train', 'val']

    for dataset_type in dataset_types:
        new_images_dir = join(processed_root, f'images/{dataset_type}')
        new_labels_dir = join(processed_root, f'labels/{dataset_type}')
        label_text_name = join(raw_root, f'wider_face_split/wider_face_{dataset_type}_bbx_gt.txt')
        images_root = join(raw_root, f'WIDER_{dataset_type}/images')

        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_labels_dir, exist_ok=True)
        if remove_processed:
            remove_tree(new_images_dir, keep_dir=keep_dir)
            remove_tree(new_labels_dir, keep_dir=keep_dir)
        annots = open(label_text_name)
        lines = annots.readlines()
        names = [x for x in lines if 'jpg' in x]
        indices = [lines.index(x) for x in names]

        print(f"Processing {dataset_type} data:")
        for n in tqdm(range(len(names[:]))):
            i = indices[n]
            name = lines[i].rstrip()
            old_img_path = os.path.join(images_root , name)
            name = name.split('/')[-1]
            label_path = os.path.join(new_labels_dir , name.split('.')[0] + '.txt')
            img_path = os.path.join(new_images_dir , name)
            
            num_objs = int(lines[i+1].rstrip())
            bboxs = lines[i+2 : i+2+num_objs]
            bboxs = list(map(lambda x:x.rstrip() , bboxs))
            bboxs = list(map(lambda x:x.split()[:4], bboxs))
            img = cv2.imread(old_img_path)
            img_h,img_w,_ = img.shape
            img_h,img_w,_ = img.shape
            f = open(label_path, 'w')
            count = 0
            for bbx in bboxs:
                x1 = int(bbx[0])
                y1 = int(bbx[1])
                w = int(bbx[2])
                h = int(bbx[3])
                x = (x1 + w//2) / img_w
                y = (y1 + h//2) / img_h
                w = w / img_w
                h = h / img_h
                if w * h * 100 > pixel_threshold:
                    yolo_line = f'{0} {x} {y} {w} {h}\n'
                    f.write(yolo_line)
                    count += 1
            f.close()
            if count > 0:   
                copy(old_img_path , img_path)
            else:
                os.remove(label_path)

    if process_test:
        new_images_dir = join(processed_root, 'images/test')
        label_text_name = join(raw_root, 'wider_face_split/wider_face_test_filelist.txt')
        images_root = join(raw_root, 'WIDER_test/images')
        
        os.makedirs(new_images_dir, exist_ok=True)
        if remove_processed and exists(processed_root):
            remove_tree(new_images_dir, keep_dir=True)
        annots = open(label_text_name)
        lines = annots.readlines()
        names = [x for x in lines if 'jpg' in x]
        indices = [lines.index(x) for x in names]

        print('Processing test data:')
        for n in tqdm(range(len(names[:]))):
            i = indices[n]
            name = lines[i].rstrip()
            old_img_path = os.path.join(images_root , name)
            name = name.split('/')[-1]
            img_path = os.path.join(new_images_dir , name)
            img = cv2.imread(old_img_path)
            copy(old_img_path , img_path)

    print("Wider Face dataset Processed!")

def preprocess_fer2013(csv_path):
    print("Preprocessing FER2013.")
    df = pd.read_csv(csv_path)
    
    emotions = {
        0:"Angry",
        1:"Disgust",
        2:"Fear",
        3:"Happy",
        4:"Sad",
        5:"Surprize",
        6:"Neutral"
    }

    X_train,y_train = [],[]
    X_val,y_val = [],[]
    X_test,y_test = [],[]
    for index, row in df.iterrows():
        k = row['pixels'].split(" ")
        if row['Usage'] == 'Training':
            X_train.append(np.array(k))
            y_train.append(row['emotion'])
        elif row['Usage'] == 'PrivateTest':
            X_test.append(np.array(k))
            y_test.append(row['emotion'])
        elif row['Usage'] == 'PublicTest':
            X_val.append(np.array(k))
            y_val.append(row['emotion'])
    
    X_train = np.array(X_train,dtype='float')
    y_train = np.array(y_train)
    X_test = np.array(X_test,dtype='float')
    y_test = np.array(y_test)
    X_val = np.array(X_val,dtype='float')
    y_val = np.array(y_val)
    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_val = X_val.reshape(X_val.shape[0], 1, 48, 48)

    processed_fer2013 = {
        'x_train': X_train,
        'x_val': X_val,
        'x_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    print("FER2013 processed!")
    return processed_fer2013

def merge_yolo(dataset_root, yolo_root, label_swap=None):
    dataset_types = ['train', 'valid']

    for dataset_type in dataset_types:
        label_roots = join(dataset_root, f"{dataset_type}/labels")
        label_filenames = listdir(label_roots)
        for n in tqdm(range(len(label_filenames))):
            label_filename = label_filenames[n]
            prefix = label_filename.split(".txt")[0]
            label_dstpath = join(yolo_root, f"labels/{dataset_type}/{label_filename}")
            
            image_filepath = join(dataset_root, f"{dataset_type}/images/{prefix}.jpg")
            image_dstpath = join(yolo_root, f"images/{dataset_type}/{prefix}.jpg")
            if not exists(image_filepath):
                image_filepath = join(dataset_root, f"{dataset_type}/images/{prefix}.png")
                image_dstpath = join(yolo_root, f"images/{dataset_type}/{prefix}.png")
            if dataset_type == 'valid':
                image_dstpath = image_dstpath.replace('valid/', 'val/')
                label_dstpath = label_dstpath.replace('valid/', 'val/')
            copy(image_filepath, image_dstpath)

            annots = open(join(label_roots, label_filename))
            lines = annots.readlines()
            f = open(label_dstpath, 'w')
            for index, line in enumerate(lines):
                if line[0] == label_swap['face']:
                    f.write(line.replace(label_swap['face'], label_swap['other'], 1))
                elif line[0] == label_swap['other']:
                    f.write(line.replace(label_swap['other'], label_swap['face'], 1))
                else:
                    f.write(line)
            f.close()
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
CSCI-4962-Projects-in-ML-and-AI/preprocess_utils.py at master · DimensionPrism/CSCI-4962-Projects-in-ML-and-AI