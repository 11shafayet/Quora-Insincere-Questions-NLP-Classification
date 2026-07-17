from pathlib import Path

from sklearn.model_selection import train_test_split

from quora_insincere.config import ModelConfig, Paths
from quora_insincere.data import load_competition_data, save_submission
from quora_insincere.features.embeddings import build_embedding_matrix, load_embeddings
from quora_insincere.features.sequences import prepare_sequences
from quora_insincere.features.text import clean_text_v2
from quora_insincere.models.lstm import build_lstm_model
from quora_insincere.utils.metrics import find_best_threshold
from quora_insincere.utils.training import make_class_weight_dict, make_early_stopping


def main():
    paths = Paths()
    config = ModelConfig()
    train_df, test_df = load_competition_data(paths)

    cleaned_questions = train_df["question_text"].apply(clean_text_v2)
    X_train_clean, X_val_clean, y_train, y_val = train_test_split(
        cleaned_questions,
        train_df["target"],
        test_size=config.validation_size,
        stratify=train_df["target"],
        random_state=config.random_state,
    )
    test_questions_clean = test_df["question_text"].apply(clean_text_v2)

    tokenizer, X_train_pad, X_val_pad, X_test_pad = prepare_sequences(
        X_train_clean, X_val_clean, test_questions_clean, config
    )
    embeddings_index = load_embeddings(paths.glove_txt)
    embedding_matrix = build_embedding_matrix(
        tokenizer, embeddings_index, config.max_vocab_size, config.embed_dim
    )

    model = build_lstm_model(
        config.max_len,
        config.max_vocab_size + 1,
        config.embed_dim,
        embedding_matrix,
    )
    model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=make_class_weight_dict(y_train),
        callbacks=[make_early_stopping()],
    )

    y_val_proba = model.predict(X_val_pad, batch_size=config.batch_size).squeeze()
    best_threshold, best_f1 = find_best_threshold(y_val, y_val_proba)
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Maximum F1-Score: {best_f1:.4f}")

    test_predictions = (model.predict(X_test_pad, batch_size=config.batch_size).squeeze() >= best_threshold).astype(int)
    save_submission(test_df, test_predictions, Path(paths.output_dir) / "lstm_submission.csv")
    Path(paths.model_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(paths.model_dir) / "lstm_model.keras")


if __name__ == "__main__":
    main()

