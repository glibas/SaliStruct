# File: metrics/attention_metrics.py

import numpy as np
import cv2
from scipy.ndimage import center_of_mass, label
from skimage.measure import shannon_entropy


def compute_attention_entropy(saliency_map):
    """
    Compute Shannon entropy of the saliency map.
    Higher entropy => more dispersed attention.
    """
    entropy = shannon_entropy(saliency_map)
    return 1 - entropy / 10


def compute_attention_clusters(saliency_map, threshold_ratio=0.6):
    """
    Count distinct attention clusters using connected components above a threshold.
    """
    threshold = int(threshold_ratio * np.max(saliency_map))
    _, binary = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    labeled, num_features = label(binary)
    return 1 / (1 + num_features)


def compute_center_of_gravity(saliency_map):
    """
    Compute center of gravity (attention centroid) normalized to image dimensions.
    """
    y, x = center_of_mass(saliency_map)
    h, w = saliency_map.shape
    norm_x = x / w
    norm_y = y / h
    return 1 - np.minimum(np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2) / 0.71, 1)


def compute_attention_metrics(saliency_map):
    return {
        "entropy": compute_attention_entropy(saliency_map),
        "clusters_score": compute_attention_clusters(saliency_map),
        "center_of_gravity": compute_center_of_gravity(saliency_map),
    }
