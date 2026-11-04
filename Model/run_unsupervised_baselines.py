"""Rerun Isolation Forest, K-means and an autoencoder on the fixed ZipZap vectors."""
import argparse
import csv
import json
import math
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output_dir", default="research_results/unsupervised")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--ae_epochs", type=int, default=100)
    parser.add_argument("--ae_batch_size", type=int, default=256)
    return parser.parse_args()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"addresses", "labels", "vectors"}
        if required.difference(data.files):
            raise ValueError("%s needs addresses, labels, and vectors" % path)
        addresses = np.asarray(data["addresses"]).reshape(-1)
        labels = np.asarray(data["labels"], dtype=np.int8).reshape(-1)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or len(addresses) != len(labels) or len(labels) != len(vectors):
        raise ValueError("Invalid account-vector dimensions in " + path)
    if len(set(addresses.tolist())) != len(addresses):
        raise ValueError("Expected one row per account in " + path)
    return addresses, labels, vectors


def validate_disjoint(*splits):
    account_sets = [set(split[0].tolist()) for split in splits]
    for i, first in enumerate(account_sets):
        if any(first & later for later in account_sets[i + 1:]):
            raise ValueError("Account IDs overlap across fixed splits")


def validation_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return float(np.inf)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    k = max(1, int(math.ceil(0.01 * len(labels))))
    top = order[:k]
    return {
        "accounts": int(len(labels)),
        "known_phishing_labels": int(labels.sum()),
        "roc_auc_observed_labels": float(roc_auc_score(labels, scores)),
        "average_precision_observed_labels": float(average_precision_score(labels, scores)),
        "precision_at_1pct_observed_labels": float(labels[top].mean()),
        "recall_at_1pct_observed_labels": float(labels[top].sum() / max(1, labels.sum())),
        "threshold_from_validation": float(threshold),
        "precision_observed_labels": float(precision_score(labels, scores >= threshold, zero_division=0)),
        "recall_observed_labels": float(recall_score(labels, scores >= threshold, zero_division=0)),
        "f1_observed_labels": float(f1_score(labels, scores >= threshold, zero_division=0)),
    }


def autoencoder(train, validation, test, seed, args):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(train.shape[1],))
    h = tf.keras.layers.Dense(32, activation="relu")(inp)
    h = tf.keras.layers.Dense(8, activation="relu")(h)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    out = tf.keras.layers.Dense(train.shape[1])(h)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(train, train, validation_data=(validation, validation),
              epochs=args.ae_epochs,
              batch_size=min(args.ae_batch_size, len(train)),
              callbacks=[stop], verbose=0, shuffle=True)

    def score(values):
        reconstruction = model.predict(values, batch_size=args.ae_batch_size, verbose=0)
        return np.mean(np.square(values - reconstruction), axis=1)
    return score


def main():
    args = arguments()
    train = load(args.train)
    validation = load(args.validation)
    test = load(args.test)
    validate_disjoint(train, validation, test)
    seeds = [int(v) for v in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for seed in seeds:
        scaler = StandardScaler().fit(train[2])
        x_train = scaler.transform(train[2]).astype(np.float32)
        x_validation = scaler.transform(validation[2]).astype(np.float32)
        x_test = scaler.transform(test[2]).astype(np.float32)

        models = {}
        forest = IsolationForest(n_estimators=args.n_estimators, contamination="auto",
                                 n_jobs=-1, random_state=seed).fit(x_train)
        models["isolation_forest"] = lambda values, fitted=forest: -fitted.score_samples(values)
        centers = KMeans(n_clusters=min(args.n_clusters, len(x_train)), n_init=20,
                         random_state=seed).fit(x_train)
        models["kmeans"] = lambda values, fitted=centers: np.min(fitted.transform(values), axis=1)
        models["autoencoder"] = autoencoder(x_train, x_validation, x_test, seed, args)

        for name, score_fn in models.items():
            val_scores = np.asarray(score_fn(x_validation)).reshape(-1)
            test_scores = np.asarray(score_fn(x_test)).reshape(-1)
            threshold = validation_threshold(validation[1], val_scores)
            result = metrics(test[1], test_scores, threshold)
            result.update(method=name, seed=seed,
                          threshold_source="validation_observed_labels",
                          fit_accounts=len(x_train),
                          note="Unsupervised fit uses all training vectors without labels; metrics use observed labels.")
            results.append(result)
            output_file = os.path.join(args.output_dir, "%s_seed%s.npz" % (name, seed))
            np.savez_compressed(output_file, addresses=test[0], y_observed=test[1],
                                scores=test_scores, validation_threshold=np.asarray(threshold))
            print("%s seed=%d AP=%.5f ROC-AUC=%.5f P@1%%=%.5f" % (
                name, seed, result["average_precision_observed_labels"],
                result["roc_auc_observed_labels"],
                result["precision_at_1pct_observed_labels"]))

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
