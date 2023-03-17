import cv2

from utils.bboxes_utils import *
from model.object_detector import Predictor

def vis_bboxes(image_path, conf_thresh=0.5):
    image = cv2.imread(image_path)
    predictor = Predictor(confidence_threshold=conf_thresh)
    bboxes, classes = predictor(image)
    bboxes, classes = bboxes.detach().cpu().numpy(), classes.detach().cpu().numpy()
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    print(classes)
    for i in range(len(bboxes)):
        if classes[i] == 0:
            image = cv2.rectangle(image, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 0, 255), 1)
    return image