import os
import torch
import pytorch_lightning as pl

from os.path import exists, join
from src.multi_models.resnet.model import EmotionClassifier, resnet

def load_yolov5(ckpt_path, yolov5_path, local_repo=True):
    print(f"Loading trained YOLOv5")
    if exists(ckpt_path):
        if local_repo:
            model = torch.hub.load(yolov5_path, 'custom', path=ckpt_path, source='local', _verbose=False)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt_path, trust_repo=True, _verbose=False)
        print("YOLOv5 loaded!")
        if torch.cuda.is_available():
            return model.cuda()
        else:
            return model
    else:
        raise Exception("Checkpoint not found!")