from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    train_csv: Path = Path("/kaggle/input/competitions/quora-insincere-questions-classification/train.csv")
    test_csv: Path = Path("/kaggle/input/competitions/quora-insincere-questions-classification/test.csv")
    embeddings_zip: Path = Path("/kaggle/input/competitions/quora-insincere-questions-classification/embeddings.zip")
    glove_txt: Path = Path("/kaggle/working/glove.840B.300d/glove.840B.300d.txt")
    lstm_input_model: Path = Path("/kaggle/input/datasets/mdshafayeturrahman/quora-lstm-model/lstm_model.keras")
    lstm_working_model: Path = Path("/kaggle/working/lstm_model.keras")
    gru_input_model: Path = Path("/kaggle/input/datasets/mdshafayeturrahman/quora-gru-attention-model/gru_attention_model.keras")
    gru_working_model: Path = Path("/kaggle/working/gru_attention_model.keras")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    validation_size: float = 0.1
    max_vocab_size: int = 50_000
    max_len: int = 50
    embed_dim: int = 300
    batch_size: int = 512
    epochs: int = 10
