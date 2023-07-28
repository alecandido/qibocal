import numpy as np


def calculate_frequencies(result1, result2, result3):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.samples, result2.samples, result3.samples]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return {f"{int(v1)}{int(v2)}{int(v3)}": cnt for (v1, v2, v3), cnt in zip(values, counts)}


def calculate_probabilities(result1, result2, result3):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.samples, result2.samples, result3.samples]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return {
        f"{int(v1)}{int(v2)}{int(v3)}": cnt / nshots for (v1, v2, v3), cnt in zip(values, counts)
    }