import os
import cv2
import pandas as pd
from metrics.attention_metrics import compute_attention_metrics
from metrics.structure_metrics import compute_structure_metrics
from metrics.placement_metrics import compute_average_placement_score
from model.usability_score import compute_usability_score
from tqdm import tqdm
from torchvision import models

from saliency.gradcam_predict import GradCAM, generate_saliency_maps


def evaluate_directory(screenshot_dir, saliency_dir, metadata_path):
    df_meta = pd.read_csv(metadata_path)

    results = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Evaluating Screens"):
        rico_id = int(row["rico_id"])
        usability_rating = row["usability_rating"]
        aesthetics_rating = row["aesthetics_rating"]
        design_quality_rating = row["design_quality_rating"]
        efficency = row["efficency"]
        learnability = row["learnability"]
        filename = f"{rico_id}.jpg"

        screenshot_path = os.path.join(screenshot_dir, filename)
        saliency_path = os.path.join(saliency_dir, filename)

        if not os.path.exists(screenshot_path) or not os.path.exists(saliency_path):
            continue

        screenshot = cv2.imread(screenshot_path)
        saliency = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        try:
            attention = compute_attention_metrics(saliency)
            structure = compute_structure_metrics(screenshot)
            placement = compute_average_placement_score(screenshot, saliency)
            score = compute_usability_score(attention, structure, placement) * 9 + 1

            results.append({
                "filename": filename,
                "predicted_score": score,
                "usability_rating": usability_rating,
                "aesthetics_rating": aesthetics_rating,
                "design_quality_rating": design_quality_rating,
                "efficency": efficency,
                "learnability": learnability,
                **attention,
                **structure,
                "placement_score": placement
            })
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    return pd.DataFrame(results)

saliency_model = 'gradcam'
generate_saliency = False

if generate_saliency:
    model = models.resnet50(pretrained=True)
    grad_cam = GradCAM(model, "layer4")
    generate_saliency_maps('../../data/uicrit/screenshots', '../data/uicrit/saliency_maps/gradcam', model, grad_cam)

# Example usage:
df = evaluate_directory(
    screenshot_dir='../../data/uicrit/screenshots',
    saliency_dir=f'../../data/uicrit/saliency_maps/{saliency_model}',
    metadata_path='../../data/uicrit/uicrit_public.csv'
)
df.to_csv(f'uicrit_eval_results_{saliency_model}.csv', index=False)
