import pytest

from src.signals import generate_signal


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [(0.8, 0.6, "UP"), (0.2, 0.6, "DOWN"), (0.5, 0.6, "HOLD"),
     (0.6, 0.6, "UP"), (0.4, 0.6, "DOWN")],
)
def test_signal_uses_confidence_for_both_classes(probability, threshold, expected):
    assert generate_signal(int(probability >= 0.5), probability, threshold) == expected


def test_signal_rejects_invalid_probability():
    with pytest.raises(ValueError, match="Probability"):
        generate_signal(1, 1.1)
