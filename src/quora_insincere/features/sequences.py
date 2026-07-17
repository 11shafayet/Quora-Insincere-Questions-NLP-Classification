from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from quora_insincere.config import ModelConfig


def prepare_sequences(X_train_clean, X_val_clean, test_questions_clean, config: ModelConfig = ModelConfig()):
    tokenizer = Tokenizer(num_words=config.max_vocab_size)
    tokenizer.fit_on_texts(X_train_clean)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train_clean),
        maxlen=config.max_len,
        padding="pre",
        truncating="pre",
    )
    X_val_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_val_clean),
        maxlen=config.max_len,
        padding="pre",
        truncating="pre",
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(test_questions_clean),
        maxlen=config.max_len,
        padding="pre",
        truncating="pre",
    )
    return tokenizer, X_train_pad, X_val_pad, X_test_pad

