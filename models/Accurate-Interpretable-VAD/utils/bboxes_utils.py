import cv2
import numpy as np

def get_objects_bboxes(img, predictor, min_area, dataset_name):
    """
    Returns bboxes of given image.
    """
    bboxes, classes = predictor(img)
    bboxes, classes = bboxes.detach().cpu().numpy(), classes.detach().cpu().numpy()
    if dataset_name == 'avenue':
        bboxes, classes = bboxes[classes == 0], classes[classes == 0]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    return bboxes[bbox_areas >= min_area], bbox_areas[bbox_areas >= min_area], classes[bbox_areas >= min_area]

def delete_overlapped_bboxes(bboxes, bbox_areas, cover_threshold, classes):
    """
    Removes bboxes which overlaps which other bbox.
    """
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    sort_idx = bbox_areas.argsort()  # Index of bboxes sorted in ascending order by area size

    keep_idx = []
    for i in range(sort_idx.size):  # calculate overlap with i-th bbox
        # Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i + 1:]])
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i + 1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i + 1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i + 1:]])
        # Calculate the intersection area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > cover_threshold]
        if num.size == 0:
            keep_idx.append(sort_idx[i])
    return bboxes[keep_idx], classes[keep_idx]

def get_foreground_bboxes(img_batch, bboxes, area_threshold, binary_threshold, gauss_mask_size):
    extend = 2
    sum_grad = 0

    img1 = cv2.GaussianBlur(img_batch[0], (gauss_mask_size, gauss_mask_size), 0)
    img2 = cv2.GaussianBlur(img_batch[1], (gauss_mask_size, gauss_mask_size), 0)
    grad = cv2.absdiff(img1, img2)
    sum_grad = grad + sum_grad
    sum_grad = cv2.threshold(sum_grad, binary_threshold, 255, cv2.THRESH_BINARY)[1]  # temporal gradient

    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x, y), (x + w, y + h), color=255, thickness=1)
        area = (w + 1) * (h + 1)
        if area > area_threshold and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y + h + extend, sum_grad.shape[0])
            final_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])

    return np.array(final_bboxes)