import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def read_records(path):
    with open(path) as f:
        if str(path).endswith(".csv"):
            return list(csv.DictReader(f))
        return [json.loads(line) for line in f if line.strip()]


def read_questions(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def is_bad(value):
    return str(value).strip().lower() in ("1", "true", "yes", "bad")


def answer_all(model, records, questions, cache_path=None, workers=4):
    cache = {}
    if cache_path and Path(cache_path).exists():
        for line in open(cache_path):
            row = json.loads(line)
            cache.setdefault(row["key"], {}).update(row["answers"])
    keys = [hashlib.sha256(json.dumps([model.name, r], sort_keys=True).encode()).hexdigest() for r in records]
    todo = [i for i, k in enumerate(keys) if any(q not in cache.get(k, {}) for q in questions)]

    def one(i):
        missing = [q for q in questions if q not in cache.get(keys[i], {})]
        return i, dict(zip(missing, model.answer(records[i], missing)))

    out = open(cache_path, "a") if cache_path else None
    with ThreadPoolExecutor(workers) as pool:
        for n, (i, answers) in enumerate(pool.map(one, todo), 1):
            cache.setdefault(keys[i], {}).update(answers)
            if out:
                out.write(json.dumps({"key": keys[i], "answers": answers}) + "\n")
                out.flush()
            print(f"\rAnswered {n}/{len(todo)} records", end="", file=sys.stderr)
    if out:
        out.close()
    if todo:
        print(file=sys.stderr)
    return np.array([[cache[k][q] for q in questions] for k in keys])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cv_auc(X, y):
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(), X, y, cv=folds, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def select(X, y, max_questions, min_gain=0.002):
    chosen, best = [], 0.5
    while len(chosen) < max_questions:
        aucs = {j: cv_auc(X[:, chosen + [j]], y) for j in range(X.shape[1]) if j not in chosen}
        if not aucs:
            break
        j = max(aucs, key=aucs.get)
        if aucs[j] - best < min_gain:
            break
        chosen.append(j)
        best = aucs[j]
    return chosen, best


def fit(model, records, labels, questions, cache_path, max_questions=10):
    X = logit(answer_all(model, records, questions, cache_path))
    mean, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / sd
    y = np.array(labels, dtype=int)
    chosen, auc = select(Z, y, max_questions)
    lr = LogisticRegression().fit(Z[:, chosen], y)
    return {
        "decision_model": model.name,
        "questions": [questions[j] for j in chosen],
        "mean": mean[chosen].tolist(),
        "sd": sd[chosen].tolist(),
        "weights": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "cv_auc": auc,
        "cv_auc_all_questions": cv_auc(Z, y),
        "records": len(y),
        "bad": int(y.sum()),
    }


def score(model, spec, records):
    P = answer_all(model, records, spec["questions"])
    Z = (logit(P) - np.array(spec["mean"])) / np.array(spec["sd"])
    s = 1 / (1 + np.exp(-(Z @ np.array(spec["weights"]) + spec["intercept"])))
    return [{"score": float(si), "answers": dict(zip(spec["questions"], p.tolist()))} for si, p in zip(s, P)]
