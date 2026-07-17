from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ModelConfig, Paths


def load_competition_data(paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)
    return train_df, test_df


def split_text_target(train_df: pd.DataFrame, config: ModelConfig = ModelConfig()):
    X = train_df["question_text"]
    y = train_df["target"]
    return train_test_split(
        X,
        y,
        test_size=config.validation_size,
        stratify=y,
        random_state=config.random_state,
    )


def save_submission(test_df: pd.DataFrame, predictions, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"qid": test_df["qid"], "prediction": predictions}).to_csv(output_path, index=False)

