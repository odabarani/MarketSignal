def generate_signal(prediction, probability, threshold=0.60):
    if probability < threshold:
        return "HOLD"
    return "UP" if prediction == 1 else "DOWN"
