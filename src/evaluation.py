"""
Fonctions utilitaires pour l'évaluation des résultats.
"""

import json
import polars as pl
from typing import List


def extract_expected_name_from_labels(is_name_str: str, tokens_str: str) -> str:
    """
    Extrait le nom attendu à partir des colonnes is_name et tokens.
    
    Args:
        is_name_str: String représentant la liste des labels (ex: "[0, 0, 1, 1]")
        tokens_str: String représentant la liste des tokens (ex: "[\"Le\", \"nom\"]")
    
    Returns:
        Le nom complet attendu ou chaîne vide si aucun nom
    """
    try:
        # Parse les listes JSON
        is_name = json.loads(is_name_str)
        tokens = json.loads(tokens_str)
        
        # Extraire les tokens marqués comme nom (is_name == 1)
        name_tokens = [token for i, token in enumerate(tokens) if i < len(is_name) and is_name[i] == 1]
        
        # Joindre les tokens pour former le nom complet
        expected_name = " ".join(name_tokens).strip()
        
        return expected_name
        
    except (json.JSONDecodeError, IndexError, ValueError):
        return ""


def create_evaluation_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Crée un dataset d'évaluation avec les noms attendus extraits des labels.
    
    Args:
        df: DataFrame avec colonnes video_name, is_name, tokens
        
    Returns:
        DataFrame avec colonnes video_name, expected_name
    """
    evaluation_data = []
    
    for row in df.iter_rows(named=True):
        video_name = row['video_name']
        expected_name = extract_expected_name_from_labels(row['is_name'], row['tokens'])
        
        evaluation_data.append({
            "video_name": video_name,
            "expected_name": expected_name
        })
    
    return pl.DataFrame(evaluation_data)


def calculate_accuracy(predictions: List[str], expected: List[str]) -> dict:
    """
    Calcule différentes métriques d'accuracy.
    
    Args:
        predictions: Liste des prédictions
        expected: Liste des résultats attendus
        
    Returns:
        Dict avec les métriques d'accuracy
    """
    if len(predictions) != len(expected):
        raise ValueError("Les listes predictions et expected doivent avoir la même taille")
    
    total = len(predictions)
    exact_matches = sum(1 for pred, exp in zip(predictions, expected) if pred.strip().lower() == exp.strip().lower())
    
    # Accuracy pour noms non-vides seulement
    non_empty_expected = [(pred, exp) for pred, exp in zip(predictions, expected) if exp.strip()]
    non_empty_exact = sum(1 for pred, exp in non_empty_expected if pred.strip().lower() == exp.strip().lower())
    
    # Accuracy pour détection (vide vs non-vide)
    detection_correct = sum(1 for pred, exp in zip(predictions, expected) 
                          if (pred.strip() == "" and exp.strip() == "") or 
                             (pred.strip() != "" and exp.strip() != ""))
    
    return {
        "exact_accuracy": exact_matches / total if total > 0 else 0,
        "detection_accuracy": detection_correct / total if total > 0 else 0,
        "name_extraction_accuracy": non_empty_exact / len(non_empty_expected) if non_empty_expected else 0,
        "total_samples": total,
        "exact_matches": exact_matches,
        "names_to_extract": len(non_empty_expected),
        "names_correctly_extracted": non_empty_exact
    }
