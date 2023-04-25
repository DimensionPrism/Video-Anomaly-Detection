import os
import cv2
import torch
import argparse
import pytorch_lightning as pl

from os.path import join, exists
from torchvision import transforms
from src.utils.ckpt_utils import load_yolov5
from src.multi_models.resnet.model import resnet, EmotionClassifier

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='./50542022061314543049_jpg.rf.930d26055dc6e3ddd1ce3603fbd7da3e.jpg', help='image path for inference')
    parser.add_argument('--yolo_ckpt_path', type=str, default="./model_results/yolo_runs/exp/weights/best.pt", help='yolov5 ckpt path')
    parser.add_argument('--yolo_path', type=str, default="./src/multi_models/yolov5", help='local yolo repo path')
    parser.add_argument('--ec_ckpt_path', type=str, default='./model_results/resnet_runs/fer2013_val_accuracy=0.697.ckpt', help='emotion classifier ckpt path')
    return parser.parse_args()


if __name__ == "__main__":
    cfg = config()
    image_prefix = cfg.image.split(".jpg")[0].split("/")[-1]
    prediction_root = join("./prediction", image_prefix)
    os.makedirs(prediction_root, exist_ok=True)

    image = cv2.imread(cfg.image)
    cv2.imwrite(join(prediction_root, f'./sample.jpg'), image)


    yolov5 = load_yolov5(ckpt_path=cfg.yolo_ckpt_path, yolov5_path=cfg.yolo_path)
    yolov5.conf = 0.6
    yolov5_results = yolov5(cfg.image)
    cropped_root = join(prediction_root, 'cropped')
    os.makedirs(cropped_root, exist_ok=True)
    faces = []
    mobiles = []
    for index, obj in enumerate(yolov5_results.crop(save=False)):
        label = obj['label'].split(" ")[0]
        cv2.imwrite(join(cropped_root, f"{index}_{label}.jpg"), obj['im'])
        if 'Mobile' in obj['label']:
            mobiles.append(obj['im'])
        if 'Face' in obj['label']:
            faces.append(obj['im'])
    f = open(join(prediction_root, f"report.txt"), 'w')
    if len(faces) == 0:
        f.writelines(f"No student detected!\n")
    elif len(faces) > 1:
        f.writelines(f"At least 2 people show in the camera!\n")
    if len(mobiles) != 0:
        f.writelines(f"Smartphone detected!\n")

    resnet50 = resnet('./model_results/resnet_weights/resnet50_scratch_weight.pkl')
    emotion_classifier = EmotionClassifier.load_from_checkpoint(checkpoint_path=cfg.ec_ckpt_path, model=resnet50, processed_fer2013=None)
    pred_emotions = []
    for face in faces:
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (48, 48))
        pred_emotions.append(emotion_classifier.inference(gray_face))
    for index, pred_emotion in enumerate(pred_emotions):
        f.writelines(f"Student {index} emotion: {pred_emotion}\n")
    f.close()
    print(f"Inference completed!")