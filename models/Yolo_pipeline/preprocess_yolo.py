import os
import argparse
from roboflow import Roboflow
from os.path import join, exists
from src.utils.preprocess_utils import preprocess_widerface, merge_yolo

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./datasets/yolo')
    parser.add_argument('--wider_face_root', type=str, default='./datasets/widerface')
    parser.add_argument('--download', type=bool, default=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    cfg = config()
    if cfg.download:
        rf = Roboflow(api_key="70ncEJ2r5iEg5AFx8Ax4")
        project = rf.workspace("yolo5-kjv0a").project("tustrain")
        dataset = project.version(8).download("yolov5", './datasets/other_obj')
    # Preprocess Wider Face dataset
    preprocess_widerface(raw_root=cfg.wider_face_root, processed_root=cfg.data, remove_processed=True, keep_dir=True)
    # merge other yolo dataset with widerface
    merge_yolo('./datasets/other_obj', cfg.data, label_swap={'face': '2', 'other': '0'})

    f = open(join(cfg.data, 'meta.yaml'), 'w')
    train_image_dir = join(cfg.data, 'images/train')
    val_image_dir = join(cfg.data, 'images/val')
    test_image_dir = join(cfg.data, 'images/test')
    f.write(f'train: {train_image_dir}')
    f.write(f'\nval: {val_image_dir}')
    f.write(f'\ntest: {test_image_dir}')
    f.write(f'\nnc: {4}')
    f.write("\nnames: ['Face', 'Clen', 'Cam', 'Mobile']")