

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
    print(device)

    sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth").to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image = cv2.imread('./test3.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    '''
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    '''

    print("gen mask")

    masks = mask_generator.generate(image)

    plt.figure(figsize=(10,10))
    plt.imshow(image)

    for i, mask in enumerate(masks):
        mask = mask["segmentation"]
        show_mask(mask, plt.gca())

    plt.axis('off')
    plt.show()  
  





