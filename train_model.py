"""Train and save a Random Forest model for packaging damage prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "logistics_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURE_COLUMNS = [
    "product_weight_kg",
    "transit_distance_km",
    "material_burst_strength_kpa",
    "fragility_score",
    "cost_inr",
]
TARGET_COLUMN = "damage_probability_pct"


def main() -> None:
    """Load the generated CSV, evaluate a held-out model, and save the model."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run dataset_generator.py first."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("The dataset is empty.")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Test MSE: {mse:.3f}")
    print(f"Test R²:  {r2:.3f}")

    # Saving feature names with the estimator makes the expected input schema
    # easy to inspect when the model is loaded for inference.
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
