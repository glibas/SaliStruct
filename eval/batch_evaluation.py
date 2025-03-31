import os
import cv2
import pandas as pd
from metrics.attention_metrics import compute_attention_metrics
from metrics.structure_metrics import compute_structure_metrics
from metrics.placement_metrics import compute_average_placement_score
from model.usability_score import compute_usability_score
from tqdm import tqdm


def evaluate_directory(screenshot_dir, saliency_dir, metadata_path):
    df_meta = pd.read_csv(metadata_path)
    df_avg = df_meta.groupby("rico_id")["usability_rating"].mean().reset_index()
    df_avg.rename(columns={"usability_rating": "avg_usability_rating"}, inplace=True)

    results = []

    for _, row in tqdm(df_avg.iterrows(), total=len(df_avg), desc="Evaluating Screens"):
        rico_id = int(row["rico_id"])
        usability_rating = row["avg_usability_rating"]
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
            score = compute_usability_score(attention, structure, placement)

            results.append({
                "filename": filename,
                "predicted_score": score,
                "usability_rating": usability_rating,
                **attention,
                **structure,
                "placement_score": placement
            })
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    return pd.DataFrame(results)

saliency_model = 'gradcam'

# Example usage:
df = evaluate_directory(
    screenshot_dir='../data/uicrit/screenshots',
    saliency_dir=f'../data/uicrit/saliency_maps/{saliency_model}',
    metadata_path='../data/uicrit/uicrit_public.csv'
)
df.to_csv(f'uicrit_eval_results_{saliency_model}.csv', index=False)
