# File: model/usability_score.py

def compute_usability_score(attention_metrics, structure_metrics, placement_score=None, weights=None):
    """
    Combine attention, structure, and placement metrics into a composite usability score.

    Parameters:
    - attention_metrics: dict with keys 'entropy', 'clusters', 'center_of_gravity'
    - structure_metrics: dict with keys 'symmetry', 'visual_coverage'
    - placement_score: float between 0 and 1
    - weights: dict or None; if None, use default weights

    Returns:
    - usability_score: float
    """
    if weights is None:
        weights = {
            'entropy': 0.15,
            'clusters': 0.1,
            'center_of_gravity': 0.1,
            'symmetry': 0.25,
            'visual_coverage': 0.25,
            'placement': 0.15
        }

    if placement_score is None:
        placement_score = 0.0

    # Normalize center of gravity distance to center (closer is better)
    cog_x = attention_metrics['center_of_gravity_x']
    cog_y = attention_metrics['center_of_gravity_y']
    cog_dist = ((cog_x - 0.5) ** 2 + (cog_y - 0.5) ** 2) ** 0.5  # range 0 to ~0.7
    normalized_cog = 1 - min(cog_dist / 0.71, 1)  # closer to center => closer to 1

    score = (
            weights['entropy'] * (1 - attention_metrics['entropy'] / 10) +  # lower entropy is better
            weights['clusters'] * (1 / (1 + attention_metrics['num_clusters'])) +
            weights['center_of_gravity'] * normalized_cog +
            weights['symmetry'] * structure_metrics['symmetry_score'] +
            weights['visual_coverage'] * structure_metrics['visual_coverage'] +
            weights['placement'] * placement_score
    )

    return round(score, 4)
