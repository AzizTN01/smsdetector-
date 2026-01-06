import argparse
import joblib


def predict_text(model_path: str, texts):
    clf = joblib.load(model_path)
    if isinstance(texts, str):
        texts = [texts]
    preds = clf.predict(texts)
    probs = None
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(texts)
    results = []
    for i, t in enumerate(texts):
        p = preds[i]
        prob_spam = None
        if probs is not None:
            # spam class is 1
            prob_spam = float(probs[i][1])
        results.append({'text': t, 'pred': int(p), 'prob_spam': prob_spam})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/spam_classifier.joblib')
    parser.add_argument('--text', type=str, help='Message text to classify')
    parser.add_argument('--file', type=str, help='File with one message per line')
    args = parser.parse_args()

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        print('No texts provided. Use --text or --file')
        raise SystemExit(1)

    results = predict_text(args.model, texts)
    for r in results:
        label = 'spam' if r['pred'] == 1 else 'ham'
        if r['prob_spam'] is not None:
            print(f"{label} (spam_prob={r['prob_spam']:.3f}): {r['text']}")
        else:
            print(f"{label}: {r['text']}")

