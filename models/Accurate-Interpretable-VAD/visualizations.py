import os
import cv2
from utils.vis_utils import vis_bboxes

os.makedirs("./vis", exist_ok=True)

image_path = "./data/custom_dataset/training/frames/1/0.jpg"
for i in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    new_image = vis_bboxes(image_path, i)
    cv2.imwrite(f"./vis/temp_{i}.jpg", new_image)