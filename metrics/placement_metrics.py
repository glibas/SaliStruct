# File: metrics/placement_metrics.py

import cv2
import numpy as np


def extract_ui_elements(image, min_area=500):
    """
    Use contour detection to extract bounding boxes of likely UI elements.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            bboxes.append((x, y, w, h))
    return bboxes


def compute_iou(boxA, boxB):
    """
    Compute intersection over union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def compute_average_placement_score(image, saliency_map):
    """
    Compute the average placement score of inferred UI elements
    based on overlap with top 25% saliency areas.
    """
    bboxes = extract_ui_elements(image)

    # Threshold saliency map to get top focus regions
    thresh_val = int(0.75 * np.max(saliency_map))
    _, high_attention = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)
    high_attention = high_attention.astype(np.uint8)
    sal_contours, _ = cv2.findContours(high_attention, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sal_bboxes = [cv2.boundingRect(c) for c in sal_contours]

    if not bboxes or not sal_bboxes:
        return 0.0

    iou_scores = []
    for ui_box in bboxes:
        best_iou = max([compute_iou(ui_box, sal_box) for sal_box in sal_bboxes])
        iou_scores.append(best_iou)

    return round(np.mean(iou_scores), 4) if iou_scores else 0.0