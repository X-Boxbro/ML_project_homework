from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "ml_train.csv")
    test_df = pd.read_csv(DATA_DIR / "ml_test.csv")
    x_train = train_df.drop(columns=["critical_temp"])
    y_train = train_df["critical_temp"]
    x_test = test_df.drop(columns=["critical_temp"])
    y_test = test_df["critical_temp"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"train shape: {train_df.shape}, test shape: {test_df.shape}")
    print(f"target mean/std in train: {y_train.mean():.3f} / {y_train.std():.3f}")
    print(f"rmse: {rmse:.4f}")
    print(f"mae: {mae:.4f}")
    print(f"r2: {r2:.4f}")


if __name__ == "__main__":
    main()
