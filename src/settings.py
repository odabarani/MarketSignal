import math


def validate_settings(threshold, retrain_days, cost_bps, slippage_bps):
    try:
        values = [float(value) for value in (threshold, retrain_days, cost_bps, slippage_bps)]
    except (TypeError, ValueError):
        raise ValueError("Complete all research settings with valid numbers.") from None
    threshold, retrain_days, cost_bps, slippage_bps = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Research settings must be finite numbers.")
    if not .5 <= threshold <= .8 or retrain_days not in (21, 42, 63):
        raise ValueError("Choose a 50–80% threshold and a listed retrain frequency.")
    if not 0 <= cost_bps <= 100 or not 0 <= slippage_bps <= 100:
        raise ValueError("Costs and slippage must be between 0 and 100 bps.")
    return threshold, int(retrain_days), cost_bps, slippage_bps
