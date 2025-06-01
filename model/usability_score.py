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
            'center_of_gravity': 0.75,
            'symmetry': 0.25,
            'visual_coverage': 0.25,
            'placement': 0.15
        }

    if placement_score is None:
        placement_score = 0.0


    score = (
            weights['entropy'] * attention_metrics['entropy'] +  # lower entropy is better
            weights['clusters'] * attention_metrics['clusters_score'] +
            weights['center_of_gravity'] * attention_metrics['center_of_gravity'] +
            weights['symmetry'] * structure_metrics['symmetry_score'] +
            weights['visual_coverage'] * structure_metrics['visual_coverage'] +
            weights['placement'] * placement_score
    )

    return round(score, 4)
