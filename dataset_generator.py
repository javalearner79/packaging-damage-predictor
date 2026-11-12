"""Generate a synthetic e-commerce packaging and damage dataset.

Run this file to create ``logistics_data.csv`` beside the script. The data is
synthetic and intended for portfolio demonstrations, not operational use.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 500
OUTPUT_PATH = Path(__file__).resolve().parent / "logistics_data.csv"


def generate_dataset(row_count: int = ROW_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create logistics features and a noisy, bounded damage-probability target."""
    rng = np.random.default_rng(seed)

    # Broad but plausible ranges for a mixed e-commerce parcel population.
    weight = rng.lognormal(mean=0.15, sigma=0.75, size=row_count).clip(0.1, 25.0)
    distance = rng.uniform(20, 2_500, size=row_count)
    burst_strength = rng.uniform(150, 1_000, size=row_count)
    fragility = rng.integers(1, 6, size=row_count)
    cost = rng.lognormal(mean=5.3, sigma=0.85, size=row_count).clip(10, 2_500)

    # Build a damage-risk score from handling load, route exposure, and package
    # protection. The sigmoid maps that score to a percentage; noise represents
    # unobserved factors such as handling conditions and packaging defects.
    load_stress = weight / (burst_strength / 250.0)
    route_exposure = distance / 1_000.0
    fragility_risk = (fragility - 1) / 4.0
    value_care = np.log1p(cost) / np.log1p(2_500)

    risk_score = (
        -3.1
        + 0.95 * load_stress
        + 1.35 * route_exposure
        + 2.05 * fragility_risk
        - 1.45 * (burst_strength / 1_000.0)
        + 0.28 * value_care
        + 0.35 * load_stress * route_exposure
    )
    base_probability = 100.0 / (1.0 + np.exp(-risk_score))
    damage_probability = np.clip(
        base_probability + rng.normal(loc=0.0, scale=5.0, size=row_count), 0.0, 100.0
    )

    return pd.DataFrame(
        {
            "product_weight_kg": np.round(weight, 3),
            "transit_distance_km": np.round(distance, 1),
            "material_burst_strength_kpa": np.round(burst_strength, 1),
            "fragility_score": fragility.astype(int),
            "cost_inr": np.round(cost, 2),
            "damage_probability_pct": np.round(damage_probability, 2),
        }
    )


def main() -> None:
    data = generate_dataset()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(data)} rows: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
