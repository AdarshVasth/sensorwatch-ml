"""Train and evaluate an Isolation Forest on synthetic machine telemetry."""

from functools import lru_cache

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

FEATURES = (
    ("Temperature (°C)", 62.0, 0, 200),
    ("Vibration (mm/s)", 2.0, 0, 50),
    ("Motor current (A)", 8.0, 0, 100),
)


def normal_readings(seed, count):
    rng = np.random.default_rng(seed)
    load = rng.normal(0, 1, count)
    return np.column_stack([
        62 + 3 * load + rng.normal(0, 1.8, count),
        2 + 0.18 * load + rng.normal(0, 0.22, count),
        8 + 0.7 * load + rng.normal(0, 0.45, count),
    ])


def benchmark_data():
    """Create a reproducible holdout that is independent of the fit data."""
    normal = normal_readings(71, 200)
    faults = normal_readings(103, 60)
    faults[:20, 0] += 12
    faults[20:40, 1] += 1.2
    faults[40:, 2] += 3.1
    return np.vstack([normal, faults]), np.repeat([0, 1], [200, 60])


@lru_cache(maxsize=1)
def train():
    baseline = normal_readings(42, 600)
    model = IsolationForest(n_estimators=120, contamination=0.05, random_state=42)
    model.fit(baseline)
    return model


def validate(values):
    if len(values) != len(FEATURES):
        raise ValueError("Enter exactly three sensor readings.")
    readings = [float(value) for value in values]
    for (name, _, lower, upper), reading in zip(FEATURES, readings):
        if not lower <= reading <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}.")
    return readings


def score(values):
    readings = validate(values)
    margin = float(-train().decision_function([readings])[0])
    return margin, margin > 0


def evaluate():
    readings, labels = benchmark_data()
    margins = -train().decision_function(readings)
    tn, fp, fn, tp = confusion_matrix(labels, margins > 0, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(labels, margins)),
        "Average precision": float(average_precision_score(labels, margins)),
        "True negatives": int(tn),
        "False positives": int(fp),
        "False negatives": int(fn),
        "True positives": int(tp),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=2))
