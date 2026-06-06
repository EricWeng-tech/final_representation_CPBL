"""Train and save the tuned Random Forest model used by the CPBL GUI."""

from __future__ import annotations

import sys

from gui_predict import (
    MODEL_PATH,
    RF_PARAMS,
    load_data,
    save_random_forest_model,
    train_random_forest,
)


def main() -> int:
    df, feature_cols = load_data()
    model, train_n, train_years = train_random_forest(df, feature_cols)
    save_random_forest_model(model, feature_cols, train_n, train_years)

    print("Saved tuned Random Forest model")
    print(f"  path: {MODEL_PATH}")
    print(f"  training games: {train_n}")
    print(f"  training seasons: {train_years}")
    print("  parameters:")
    for key, value in RF_PARAMS.items():
        print(f"    {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
