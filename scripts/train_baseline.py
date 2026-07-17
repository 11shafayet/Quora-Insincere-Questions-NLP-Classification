from pathlib import Path

from sklearn.metrics import classification_report

from quora_insincere.config import Paths
from quora_insincere.data import load_competition_data, save_submission, split_text_target
from quora_insincere.models.baseline_tfidf import build_model, build_vectorizer
from quora_insincere.utils.metrics import find_best_threshold


def main():
    paths = Paths()
    train_df, test_df = load_competition_data(paths)
    X_train, X_val, y_train, y_val = split_text_target(train_df)

    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = build_model()
    model.fit(X_train_tfidf, y_train)

    y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, y_val_proba)
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Maximum F1-Score: {best_f1:.4f}")
    print(classification_report(y_val, (y_val_proba >= best_threshold).astype(int)))

    X_test_tfidf = vectorizer.transform(test_df["question_text"])
    test_predictions = (model.predict_proba(X_test_tfidf)[:, 1] >= best_threshold).astype(int)
    save_submission(test_df, test_predictions, Path(paths.output_dir) / "baseline_submission.csv")


if __name__ == "__main__":
    main()

