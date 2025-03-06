import numpy as np

def calculate_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)