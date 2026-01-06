import argparse
import os
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import download_dataset, load_raw
from .preprocess import prepare_dataframe


def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    return pipe


def train(data_path: str, model_path: str, test_size: float = 0.2, random_state: int = 42):
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, downloading...")
        download_dataset(data_path)

    df_raw = load_raw(data_path)
    df = prepare_dataframe(df_raw)
    X = df['text']
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    metrics = {
        'classification_report': report,
        'confusion_matrix': cm
    }
    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/SMSSpamCollection')
    parser.add_argument('--model', type=str, default='models/spam_classifier.joblib')
    args = parser.parse_args()
    train(args.data, args.model)
