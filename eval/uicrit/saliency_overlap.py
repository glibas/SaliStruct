# Re-import necessary packages after code state reset
import os
import json
import pandas as pd
import numpy as np
import cv2

# Re-define all required functions
def load_saliency_map(saliency_path):
    sal_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
    return sal_map / 255.0  # Normalize to [0, 1]

def compute_saliency_in_box(sal_map, bbox, image_shape):
    h, w = image_shape
    x1, y1, x2, y2 = [int(b * dim) for b, dim in zip(bbox, [w, h, w, h])]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    box_region = sal_map[y1:y2, x1:x2]
    if box_region.size == 0:
        return 0.0
    return np.mean(box_region)

def extract_bounding_boxes(comment_str):
    bboxes = []
    lines = comment_str.split("Bounding Box:")
    for part in lines[1:]:
        bbox_str = part.strip().split("]")[0] + "]"
        try:
            bbox = json.loads(bbox_str)
            bboxes.append(bbox)
        except json.JSONDecodeError:
            continue
    return bboxes

def analyze_saliency_overlap_with_sources(csv_path, saliency_dir, image_shape=(1920, 1080)):
    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        rico_id = str(row['rico_id'])
        sal_map_path = os.path.join(saliency_dir, f"{rico_id}.jpg")
        if not os.path.exists(sal_map_path):
            continue

        sal_map = load_saliency_map(sal_map_path)
        comment_str = row['comments']
        sources_str = row['comments_source']
        bboxes = extract_bounding_boxes(comment_str)

        # Parse sources list
        try:
            sources = eval(sources_str) if isinstance(sources_str, str) else []
        except:
            sources = ['unknown'] * len(bboxes)

        if len(sources) != len(bboxes):
            sources = ['unknown'] * len(bboxes)

        for bbox, source in zip(bboxes, sources):
            avg_sal = compute_saliency_in_box(sal_map, bbox, image_shape)
            results.append({
                'rico_id': rico_id,
                'avg_saliency': avg_sal,
                'bbox': bbox,
                'comment_source': source
            })

    return pd.DataFrame(results)

# Sample usage:
result_df = analyze_saliency_overlap_with_sources("../../data/uicrit/uicrit_public.csv", "../../data/uicrit/saliency_maps/gradcam")
result_df.to_csv("bbox_saliency_analysis.csv", index=False)
result_df.head()
