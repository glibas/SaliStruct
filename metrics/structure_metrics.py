# File: metrics/structure_metrics.py

import cv2
import numpy as np


def compute_symmetry_score(image):
    """
    Estimate horizontal symmetry by comparing left and right halves of the image.
    1 = perfect symmetry, 0 = total asymmetry.
    """
    h, w = image.shape[:2]
    mid = w // 2
    left = image[:, :mid]
    right = cv2.flip(image[:, mid:], 1)
    min_width = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:, :min_width].astype('float') - right[:, :min_width].astype('float'))
    score = 1 - (np.mean(diff) / 255.0)
    return np.clip(score, 0, 1)


def compute_visual_coverage(image):
    """
    Estimate visual coverage as the percentage of non-background pixels.
    Works best when UI elements have clear contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coverage = np.sum(binary > 0) / binary.size
    return round(coverage, 4)


def compute_structure_metrics(image):
    return {
        "symmetry_score": compute_symmetry_score(image),
        "visual_coverage": compute_visual_coverage(image)
    }
