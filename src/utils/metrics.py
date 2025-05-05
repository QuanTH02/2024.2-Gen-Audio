import numpy as np
from typing import List
import jiwer

def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate (WER) between predictions and references using jiwer.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
    
    Returns:
        float: Average WER across all samples
    """
    # Join all predictions and references into single strings
    predictions_text = " ".join(predictions)
    references_text = " ".join(references)
    
    # Calculate WER using jiwer
    wer = jiwer.wer(references_text, predictions_text)
    
    return wer

def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate (CER) between predictions and references.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
    
    Returns:
        float: Average CER across all samples
    """
    # Join all predictions and references into single strings
    predictions_text = " ".join(predictions)
    references_text = " ".join(references)
    
    # Calculate CER using jiwer
    cer = jiwer.cer(references_text, predictions_text)
    
    return cer

def calculate_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match accuracy between predictions and references.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
    
    Returns:
        float: Accuracy (percentage of exact matches)
    """
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(predictions) if predictions else 0.0

def get_wer_components(predictions: List[str], references: List[str]) -> dict:
    """
    Get detailed WER components (substitutions, deletions, insertions).
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
    
    Returns:
        dict: Dictionary containing WER components
    """
    # Join all predictions and references into single strings
    predictions_text = " ".join(predictions)
    references_text = " ".join(references)
    
    # Get WER components using jiwer
    measures = jiwer.compute_measures(references_text, predictions_text)
    
    return {
        "wer": measures["wer"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "total_words": measures["words"]
    } 