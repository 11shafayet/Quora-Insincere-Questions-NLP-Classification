# Quora Insincere Questions: NLP Classification

Kaggle notebook link: 
`https://www.kaggle.com/code/mdshafayeturrahman/quora-insincere-questions-nlp-classification`
Clean project version of the Kaggle notebook:

`notebooks/quora-insincere-questions-nlp-classification.ipynb`

The notebook is copied exactly from the attached updated notebook and kept as the source narrative. The reusable project code is split into small modules under `src/`.

## Structure

```text
.
├── notebooks/
│   └── quora-insincere-questions-nlp-classification.ipynb
├── scripts/
│   ├── train_baseline.py
│   ├── train_lstm.py
│   └── train_gru_attention.py
├── src/quora_insincere/
│   ├── config.py
│   ├── data.py
│   ├── features/
│   │   ├── embeddings.py
│   │   ├── sequences.py
│   │   └── text.py
│   ├── models/
│   │   ├── baseline_tfidf.py
│   │   ├── lstm.py
│   │   └── gru_attention.py
│   └── utils/
│       ├── metrics.py
│       └── training.py
└── pyproject.toml
```

## Models

- `baseline_tfidf.py`: TF-IDF + Logistic Regression baseline.
- `lstm.py`: Bidirectional LSTM with frozen GloVe embeddings.
- `gru_attention.py`: Bidirectional GRU with custom attention pooling.

## Run

Install the package in editable mode:

```bash
pip install -e .
```

Run one model:

```bash
python scripts/train_baseline.py
python scripts/train_lstm.py
python scripts/train_gru_attention.py
```

The defaults target Kaggle competition paths. For local runs, update `src/quora_insincere/config.py` or pass equivalent paths in your own wrapper.

