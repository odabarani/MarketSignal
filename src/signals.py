def generate_signal(prediction, probability, threshold=0.60):
    """Return UP, DOWN, or HOLD from the model's probability of UP."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if not 0.5 <= threshold <= 1:
        raise ValueError("Threshold must be between 0.5 and 1.")

    if probability >= threshold:
        return "UP"
    if 1 - probability >= threshold:
        return "DOWN"
    return "HOLD"
