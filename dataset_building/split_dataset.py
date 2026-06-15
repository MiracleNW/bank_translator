from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "dataset.csv"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "val.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл dataset.csv не найден: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH).dropna()

    required_columns = {"instruction", "input", "output"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В {INPUT_PATH} нет колонок: {', '.join(sorted(missing))}")

    if len(df) < 3:
        raise ValueError("Для разделения на train/val/test нужно минимум 3 строки в dataset.csv")

    print(f"Всего строк в dataset: {len(df)}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
    )

    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
