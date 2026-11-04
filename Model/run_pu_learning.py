"""Train account-level BCE and nnPU heads on fixed ZipZap vectors.

Input NPZ files must each contain one row per account with ``addresses``,
``labels`` and ``vectors`` arrays. Training labels are used only to form P
(known phishing) and U (all remaining accounts); validation/test labels are
used only for evaluation and validation threshold selection.
"""
import argparse
import csv
import json
import math
import os
import random

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow as tf
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training account vectors NPZ")
    parser.add_argument("--validation", required=True, help="Validation account vectors NPZ")
    parser.add_argument("--test", required=True, help="Test account vectors NPZ")
    parser.add_argument("--output_dir", default="research_results/pu")
    parser.add_argument("--model_dir", default=os.environ.get("SM_MODEL_DIR", ""),
                        help="Optional SageMaker model artifact directory; exports loadable Keras models here")
    parser.add_argument("--priors", default="0.0087,0.015,0.03,0.05",
                        help="Comma-separated phishing class priors; the training-only estimate is always added")
    parser.add_argument("--label_fractions", default="1.0,0.5,0.25",
                        help="Fractions of training positives retained as labelled P")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="nnPU negative-risk correction threshold")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="nnPU correction weight")
    return parser.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"addresses", "labels", "vectors"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("%s missing fields %s" % (path, sorted(missing)))
        addresses = np.asarray(data["addresses"]).reshape(-1)
        labels = np.asarray(data["labels"], dtype=np.int8).reshape(-1)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or len(addresses) != len(labels) or len(labels) != len(vectors):
        raise ValueError("Invalid account/vector dimensions in " + path)
    if not np.isfinite(vectors).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Non-finite vectors or non-binary labels in " + path)
    if len(set(addresses.tolist())) != len(addresses):
        raise ValueError("Each NPZ must contain one row per account: " + path)
    return addresses, labels, vectors


def check_disjoint(train, validation, test):
    sets = [set(x.tolist()) for x in (train[0], validation[0], test[0])]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Account IDs overlap across train/validation/test files")


def make_mlp(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(input_dim,), name="account_vector")
    f1 = tf.keras.layers.Dense(128, activation="relu", name="f1")(inputs)
    f2 = tf.keras.layers.Dense(128, activation="relu", name="f2")(f1)
    # Match the existing ZipZap head: the second layer has a residual from f1.
    logits = tf.keras.layers.Dense(1, name="logit")(f2 + f1)
    return tf.keras.Model(inputs, logits)


def estimate_prior_en(labels, vectors, seed):
    """Elkan-Noto selected-at-random estimate, using training data only.

    The estimate is explicitly assumption-dependent: known positives must be
    selected roughly at random among true positives. It is reported as a
    diagnostic and added to the sensitivity grid, not treated as ground truth.
    """
    labels = np.asarray(labels, dtype=np.int8)
    if min(int(labels.sum()), int((labels == 0).sum())) < 3:
        raise ValueError("At least three known positives and three U accounts are needed to estimate the prior")
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=np.float64)
    for fit_idx, hold_idx in folds.split(vectors, labels):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=seed))
        estimator.fit(vectors[fit_idx], labels[fit_idx])
        oof[hold_idx] = estimator.predict_proba(vectors[hold_idx])[:, 1]
    positive_mean = float(oof[labels == 1].mean())
    unlabeled_mean = float(oof[labels == 0].mean())
    observed_rate = float(labels.mean())
    estimate = unlabeled_mean / max(positive_mean, 1e-8)
    return float(np.clip(estimate, observed_rate, 0.99)), {
        "method": "Elkan-Noto cross-fitted label-propensity ratio",
        "assumption": "positives are selected at random for labelling",
        "mean_label_propensity_on_known_positives": positive_mean,
        "mean_label_propensity_on_unlabelled": unlabeled_mean,
        "observed_training_label_rate_lower_bound": observed_rate,
    }


def fit_model(model, positive, unlabeled, method, prior, args, seed):
    rng = np.random.default_rng(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    p = tf.convert_to_tensor(positive, tf.float32)
    u = tf.convert_to_tensor(unlabeled, tf.float32)
    p_steps = max(1, math.ceil(len(positive) / args.batch_size))
    u_steps = max(1, math.ceil(len(unlabeled) / args.batch_size))

    @tf.function
    def step(pb, ub):
        with tf.GradientTape() as tape:
            fp = tf.reshape(model(pb, training=True), [-1])
            fu = tf.reshape(model(ub, training=True), [-1])
            if method == "bce":
                positive_weight = len(positive) / float(len(positive) + len(unlabeled))
                loss = (tf.reduce_mean(tf.nn.softplus(-fp)) * positive_weight +
                        tf.reduce_mean(tf.nn.softplus(fu)) * (1.0 - positive_weight))
            else:
                positive_risk = prior * tf.reduce_mean(tf.nn.softplus(-fp))
                negative_risk = (tf.reduce_mean(tf.nn.softplus(fu)) -
                                 prior * tf.reduce_mean(tf.nn.softplus(fp)))
                # Kiryo et al. nnPU correction: reverse the gradient when the
                # estimated negative risk falls below -beta.
                loss = tf.cond(
                    negative_risk < -args.beta,
                    lambda: positive_risk - args.gamma * negative_risk,
                    lambda: positive_risk + negative_risk)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for _ in range(args.epochs):
        p_order = rng.permutation(len(positive))
        u_order = rng.permutation(len(unlabeled))
        for batch in range(max(p_steps, u_steps)):
            pi = p_order[(batch * args.batch_size) % len(p_order):
                         (batch * args.batch_size) % len(p_order) + args.batch_size]
            ui = u_order[(batch * args.batch_size) % len(u_order):
                         (batch * args.batch_size) % len(u_order) + args.batch_size]
            if not len(pi):
                pi = p_order[:args.batch_size]
            if not len(ui):
                ui = u_order[:args.batch_size]
            step(tf.gather(p, pi), tf.gather(u, ui))
    return model


def validation_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def observed_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    k = max(1, int(math.ceil(0.01 * len(labels))))
    top = order[:k]
    threshold_reports = []
    for fixed_threshold in np.arange(0.05, 1.0, 0.05):
        selected = scores >= fixed_threshold
        threshold_reports.append({
            "threshold": round(float(fixed_threshold), 2),
            "flagged_accounts": int(selected.sum()),
            "precision_observed_labels": float(precision_score(labels, selected, zero_division=0)),
            "recall_observed_labels": float(recall_score(labels, selected, zero_division=0)),
            "f1_observed_labels": float(f1_score(labels, selected, zero_division=0)),
        })
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
        "fixed_threshold_reports": threshold_reports,
    }


def main():
    args = arguments()
    train = load_npz(args.train)
    validation = load_npz(args.validation)
    test = load_npz(args.test)
    check_disjoint(train, validation, test)
    if train[1].sum() == 0 or train[1].sum() == len(train[1]):
        raise ValueError("Training split must contain both labelled P and U accounts")
    priors = [float(v) for v in args.priors.split(",")]
    fractions = [float(v) for v in args.label_fractions.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
    results = []
    prior_estimates = []

    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Label fractions must be in (0, 1]")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            train_labels = train[1]
            positive_ids = np.flatnonzero(train_labels == 1)
            keep_n = max(1, int(round(len(positive_ids) * fraction)))
            kept = set(rng.choice(positive_ids, size=keep_n, replace=False).tolist())
            # Hidden positives are moved into U; no training class-0 labels are
            # used as verified negatives.
            p_mask = np.array([i in kept for i in range(len(train_labels))])
            positive = train[2][p_mask]
            unlabeled = train[2][~p_mask]
            if not len(unlabeled):
                raise ValueError("No unlabelled accounts remain")
            available_training_labels = p_mask.astype(np.int8)
            observed_rate = float(available_training_labels.mean())
            prior_estimate, prior_estimate_details = estimate_prior_en(
                available_training_labels, train[2], seed)
            prior_estimates.append({
                "labelled_positive_fraction": fraction, "seed": seed,
                "estimated_class_prior": prior_estimate,
                "estimator_details": prior_estimate_details,
            })
            if any(not 0 < p < 1 for p in priors):
                raise ValueError("Class priors must be in (0, 1)")
            # The observed label rate is a lower bound for the class prior.
            # Clamp a rounded default (e.g. 0.0087) to the exact training rate
            # so rounding cannot abort a SageMaker job.
            sensitivity_priors = sorted(set(
                [max(p, observed_rate) for p in priors] + [prior_estimate]))

            for method in ("nnpu", "bce"):
                method_priors = sensitivity_priors if method == "nnpu" else [None]
                for prior in method_priors:
                    tf.keras.backend.clear_session()
                    model = make_mlp(train[2].shape[1], seed)
                    fit_model(model, positive, unlabeled, method,
                              prior if prior is not None else 0.0, args, seed)
                    val_scores = tf.sigmoid(model(validation[2], training=False)).numpy().reshape(-1)
                    test_scores = tf.sigmoid(model(test[2], training=False)).numpy().reshape(-1)
                    threshold = validation_threshold(validation[1], val_scores)
                    metrics = observed_metrics(test[1], test_scores, threshold)
                    metrics.update(method=method, prior=prior,
                                   labelled_positive_fraction=fraction, seed=seed,
                                   train_accounts=len(train_labels),
                                   labelled_positive_accounts=len(positive),
                                   unlabeled_accounts=len(unlabeled),
                                   training_observed_label_rate=observed_rate,
                                   estimated_class_prior=prior_estimate,
                                   prior_estimator=prior_estimate_details,
                                   prior_estimate_assumption="selected at random; may be violated by report-driven labels")
                    results.append(metrics)
                    prior_label = "baseline" if prior is None else str(prior).replace(".", "p")
                    name = "%s_pi%s_labels%s_seed%s.npz" % (
                        method, prior_label,
                        str(fraction).replace(".", "p"), seed)
                    np.savez_compressed(
                        os.path.join(args.output_dir, name),
                        addresses=test[0], y_observed=test[1], scores=test_scores,
                        validation_threshold=np.asarray(threshold))
                    weights_name = name.replace(".npz", "_weights.npz")
                    np.savez_compressed(
                        os.path.join(args.output_dir, weights_name),
                        **{"weight_%02d" % i: weight
                           for i, weight in enumerate(model.get_weights())})
                    if args.model_dir:
                        # A native Keras model is directly loadable locally and
                        # is automatically collected by SageMaker from SM_MODEL_DIR.
                        model.save(os.path.join(args.model_dir, name.replace(".npz", ".keras")))
                        with open(os.path.join(args.model_dir, name.replace(".npz", "_metadata.json")),
                                  "w", encoding="utf-8") as f:
                            json.dump({
                                "method": method, "prior": prior,
                                "labelled_positive_fraction": fraction,
                                "seed": seed, "input": "account_vector",
                                "input_width": int(train[2].shape[1]),
                                "output": "logit; apply sigmoid for score",
                                "validation_threshold": threshold,
                                "base_encoder_checkpoint": "ckpt_dir/model_72402",
                            }, f, indent=2)
                    prior_display = "baseline" if prior is None else "%.4g" % prior
                    print("%s pi=%s labels=%.2f seed=%d AP=%.5f ROC-AUC=%.5f P@1%%=%.5f" % (
                        method, prior_display, fraction, seed,
                        metrics["average_precision_observed_labels"],
                        metrics["roc_auc_observed_labels"],
                        metrics["precision_at_1pct_observed_labels"]))

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"prior_estimates_by_label_fraction_and_seed": prior_estimates,
                   "results": results}, f, indent=2)
    if results:
        with open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        threshold_rows = []
        for result in results:
            for report in result["fixed_threshold_reports"]:
                threshold_rows.append({
                    "method": result["method"], "prior": result["prior"],
                    "labelled_positive_fraction": result["labelled_positive_fraction"],
                    "seed": result["seed"], **report,
                })
        with open(os.path.join(args.output_dir, "threshold_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
            writer.writeheader()
            writer.writerows(threshold_rows)


if __name__ == "__main__":
    main()
