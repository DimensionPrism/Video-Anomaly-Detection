"""
The code in this file is adapted from:
https://github.com/LiUzHiAn/hf2vad/blob/master/pre_process/extract_bboxes.py
"""
import os
import sys
import subprocess

from tqdm import tqdm
from utils.bboxes_utils import *
from model.object_detector import Predictor
from model.video_dataset import VideoDataset
from utils.img_utils import img_tensor2numpy, img_batch_tensor2numpy

DATASET_CFGS = {
    "ped2": {"confidence_threshold": 0.5, "min_area": 10 * 10, "cover_threshold": 0.6, "binary_threshold": 18,
             "gauss_mask_size": 3, 'contour_min_area': 10 * 10},
    "avenue": {"confidence_threshold": 0.8, "min_area": 20 * 20, "cover_threshold": 0.6, "binary_threshold": 18,
               "gauss_mask_size": 5, 'contour_min_area': 40 * 40},
    "shanghaitech": {"confidence_threshold": 0.8, "min_area": 8 * 8, "cover_threshold": 0.65, "binary_threshold": 15,
                     "gauss_mask_size": 5, 'contour_min_area': 40 * 40}
}


def install_detectron2():
    print("installing detectron2...")
    p = subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git', "--quiet"])
    print("detectron2 installed!")

def download_pretrained():
    print("downloading pretrained ResNet50-FPN")
    if not os.path.exists("./model/pre_trained/model_final_f10217.pkl"):
        p = subprocess.run(["wget", "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl", "-P", "./model/pre_trained/"])
        print("ResNet50-FPN downloaded!")
    else:
        print("pretrained ResNet50-FPN exists!")

def extract_bboxes(dataset_name, dataset_root):
    # extract bboxes for train data
    dataset = VideoDataset(dataset_name=dataset_name, root=dataset_root, train=True, sequence_length=1,
                           bboxes_extractions=True)
    MIN_AREA = DATASET_CFGS[dataset_name]["min_area"]
    COVER_THRESHOLD = DATASET_CFGS[dataset_name]["cover_threshold"]
    area_threshold = DATASET_CFGS[dataset_name]["contour_min_area"]
    binary_threshold = DATASET_CFGS[dataset_name]["binary_threshold"]
    gauss_mask_size = DATASET_CFGS[dataset_name]["gauss_mask_size"]
    confidence_threshold = DATASET_CFGS[dataset_name]['confidence_threshold']
    predictor = Predictor(confidence_threshold=confidence_threshold)
    all_bboxes = []
    all_classes = []
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        batch, _ = dataset.__getitem__(idx)
        # main frame
        cur_img = img_tensor2numpy(batch[1])
        obj_bboxes, bbox_areas, classes = get_objects_bboxes(cur_img, predictor, MIN_AREA, dataset_name)

        # filter some overlapped bbox
        obj_bboxes_after_overlap_removal, classes_after_removal = delete_overlapped_bboxes(obj_bboxes, bbox_areas, COVER_THRESHOLD, classes)

        foreground_bboxes = get_foreground_bboxes(img_batch_tensor2numpy(batch), obj_bboxes_after_overlap_removal,
                                                  area_threshold, binary_threshold, gauss_mask_size)
        if foreground_bboxes.shape[0] > 0:
            cur_bboxes = np.concatenate((obj_bboxes_after_overlap_removal, foreground_bboxes), axis=0)
            cur_classes = np.concatenate((classes_after_removal, (np.zeros(len(foreground_bboxes)))), axis=0)
        else:
            cur_bboxes = obj_bboxes_after_overlap_removal
            cur_classes = classes_after_removal
        all_bboxes.append(cur_bboxes)
        all_classes.append(cur_classes)
    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                            '%s_bboxes_train.npy' % dataset_name), all_bboxes)
    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                            '%s_bboxes_train_classes.npy' % dataset_name), all_classes)

    # extract bboxes for test data
    dataset = VideoDataset(dataset_name=dataset_name, root=dataset_root, train=False, sequence_length=1,
                           bboxes_extractions=True)
    MIN_AREA = DATASET_CFGS[dataset_name]["min_area"]
    COVER_THRESHOLD = DATASET_CFGS[dataset_name]["cover_threshold"]
    area_threshold = DATASET_CFGS[dataset_name]["contour_min_area"]
    binary_threshold = DATASET_CFGS[dataset_name]["binary_threshold"]
    gauss_mask_size = DATASET_CFGS[dataset_name]["gauss_mask_size"]
    confidence_threshold = DATASET_CFGS[dataset_name]['confidence_threshold']
    predictor = Predictor(confidence_threshold=confidence_threshold)
    all_bboxes = []
    all_classes = []
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        batch, _ = dataset.__getitem__(idx)
        # main frame
        cur_img = img_tensor2numpy(batch[1])
        obj_bboxes, bbox_areas, classes = get_objects_bboxes(cur_img, predictor, MIN_AREA, dataset_name)

        # filter some overlapped bbox
        obj_bboxes_after_overlap_removal, classes_after_removal = delete_overlapped_bboxes(obj_bboxes, bbox_areas, COVER_THRESHOLD, classes)

        foreground_bboxes = get_foreground_bboxes(img_batch_tensor2numpy(batch), obj_bboxes_after_overlap_removal,
                                                  area_threshold, binary_threshold, gauss_mask_size)
        if foreground_bboxes.shape[0] > 0:
            cur_bboxes = np.concatenate((obj_bboxes_after_overlap_removal, foreground_bboxes), axis=0)
            cur_classes = np.concatenate((classes_after_removal, (np.zeros(len(foreground_bboxes)))), axis=0)
        else:
            cur_bboxes = obj_bboxes_after_overlap_removal
            cur_classes = classes_after_removal
        all_bboxes.append(cur_bboxes)
        all_classes.append(cur_classes)
    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                            '%s_bboxes_test.npy' % dataset_name), all_bboxes)
    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                            '%s_bboxes_test_classes.npy' % dataset_name), all_classes)
    print('bboxes saved!')