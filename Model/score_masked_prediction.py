"""Score fixed account sequences by pretrained ZipZap masked-token loss.

This is inference only. It never reads labels to construct scores. It reports
observed-label ranking metrics because test label 0 means unreported/unlabelled.
"""
import argparse
import csv
import json
import os
import pickle
import glob
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import average_precision_score, roc_auc_score

import modeling


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", default="./data/finetune_test.tfrecord")
    parser.add_argument("--checkpoint", default="./ckpt_dir/model_72402",
                        help="Original pretrained ZipZap checkpoint prefix")
    parser.add_argument("--config", default="./zipzap_config.json")
    parser.add_argument("--vocab", default="./data/vocab")
    parser.add_argument("--output_dir", default="research_results/masked_prediction")
    parser.add_argument("--package_dir", default=os.environ.get("SM_MODEL_DIR", ""),
                        help="Optional SageMaker model artifact directory; copies the pretrained checkpoint and inference assets")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_count", type=int, default=8,
                        help="Number of randomly selected transaction tokens scored per window")
    parser.add_argument("--mask_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--negative_samples", type=int, default=5000,
                        help="ZipZap's training-time negative-sampling count")
    parser.add_argument("--max_records", type=int, default=0,
                        help="Optional diagnostic cap; 0 processes the complete split")
    return parser.parse_args()


def read_records(path, max_records=0):
    iterator = tf.python_io.tf_record_iterator(path)
    for index, serialized in enumerate(iterator):
        if max_records and index >= max_records:
            break
        example = tf.train.Example.FromString(serialized)
        f = example.features.feature
        get = lambda key: np.asarray(f[key].int64_list.value, dtype=np.int32)
        yield {
            "address": int(get("address")[0]),
            "label": int(round(f["label"].float_list.value[0])),
            "input_ids": get("input_ids"),
            "input_positions": get("input_positions"),
            "input_counts": get("input_counts"),
            "input_mask": get("input_mask"),
            "input_io_flags": get("input_io_flags"),
            "input_values": get("input_values"),
        }


def graph_for_scoring(config_path, vocab, negative_samples):
    config = modeling.BertConfig.from_json_file(config_path)
    # Actual finetune examples use max_seq_length=100; placeholders accept the
    # selected input's length and the model config supports it.
    placeholders = {
        "input_ids": tf.placeholder(tf.int32, [None, None], name="input_ids"),
        "input_positions": tf.placeholder(tf.int32, [None, None], name="input_positions"),
        "input_counts": tf.placeholder(tf.int32, [None, None], name="input_counts"),
        "input_mask": tf.placeholder(tf.int32, [None, None], name="input_mask"),
        "input_io_flags": tf.placeholder(tf.int32, [None, None], name="input_io_flags"),
        "input_values": tf.placeholder(tf.int32, [None, None], name="input_values"),
        "masked_positions": tf.placeholder(tf.int32, [None, None], name="masked_positions"),
        "masked_ids": tf.placeholder(tf.int32, [None, None], name="masked_ids"),
        "masked_weights": tf.placeholder(tf.float32, [None, None], name="masked_weights"),
    }
    model = modeling.BertModel(
        config=config, is_training=False,
        input_ids=placeholders["input_ids"],
        input_positions=placeholders["input_positions"],
        input_io_flags=placeholders["input_io_flags"],
        input_amounts=placeholders["input_values"],
        input_counts=placeholders["input_counts"],
        input_mask=placeholders["input_mask"],
        token_type_ids=None, use_one_hot_embeddings=False,
        cross_share=run_pretrain.FLAGS.cross_share)
    embedding_table_list, factorize_table_list = model.get_embedding_table()
    run_pretrain.FLAGS.neg_sample_num = negative_samples
    run_pretrain.FLAGS.neg_strategy = "zip"
    _, per_token_loss, _ = run_pretrain.get_masked_lm_output_negative_sampling(
        config, model.get_sequence_output(), embedding_table_list,
        factorize_table_list, placeholders["masked_positions"],
        placeholders["masked_ids"], placeholders["masked_weights"], vocab)
    shape = tf.shape(placeholders["masked_weights"])
    token_loss = tf.reshape(per_token_loss, shape)
    row_scores = tf.reduce_sum(token_loss * placeholders["masked_weights"], axis=1)
    row_scores /= tf.maximum(tf.reduce_sum(placeholders["masked_weights"], axis=1), 1.0)
    return placeholders, row_scores, config


def prepare_batch(records, mask_count, seed):
    rng = np.random.default_rng(seed)
    original = np.stack([r["input_ids"] for r in records]).astype(np.int32)
    masked = original.copy()
    positions = np.zeros((len(records), mask_count), dtype=np.int32)
    labels = np.zeros((len(records), mask_count), dtype=np.int32)
    weights = np.zeros((len(records), mask_count), dtype=np.float32)
    for row, record in enumerate(records):
        # Skip slot 0 (the focal account token) to score transaction-context
        # predictability rather than account-ID frequency. Ignore special IDs.
        valid = np.flatnonzero((record["input_mask"] > 0) &
                               (np.arange(len(original[row])) > 0) &
                               (record["input_ids"] > 3))
        if not len(valid):
            continue
        count = min(mask_count, len(valid))
        selected = np.sort(rng.choice(valid, size=count, replace=False))
        positions[row, :count] = selected
        labels[row, :count] = original[row, selected]
        weights[row, :count] = 1.0
        masked[row, selected] = 1  # ZipZap [MASK] has vocab ID 1.
    features = {
        "input_ids": masked,
        "input_positions": np.stack([r["input_positions"] for r in records]).astype(np.int32),
        "input_counts": np.stack([r["input_counts"] for r in records]).astype(np.int32),
        "input_mask": np.stack([r["input_mask"] for r in records]).astype(np.int32),
        "input_io_flags": np.stack([r["input_io_flags"] for r in records]).astype(np.int32),
        "input_values": np.stack([r["input_values"] for r in records]).astype(np.int32),
        "masked_positions": positions,
        "masked_ids": labels,
        "masked_weights": weights,
    }
    return features


def main():
    args = arguments()
    # run_pretrain uses global TF1 flags. Its negative-sampling output helper is
    # reused below, so discard the already-parsed argparse options before import.
    sys.argv[:] = [sys.argv[0]]
    import run_pretrain
    globals()["run_pretrain"] = run_pretrain
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(args.input_file)
    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    if args.package_dir:
        os.makedirs(args.package_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.package_dir, "ckpt_dir")
        os.makedirs(checkpoint_dir, exist_ok=True)
        copied = []
        for source in glob.glob(args.checkpoint + ".*"):
            target = os.path.join(checkpoint_dir, os.path.basename(source))
            shutil.copy2(source, target)
            copied.append(os.path.basename(source))
        if not copied:
            raise FileNotFoundError("No checkpoint files found for prefix " + args.checkpoint)
        shutil.copy2(args.config, os.path.join(args.package_dir, "zipzap_config.json"))
        shutil.copy2(args.vocab, os.path.join(args.package_dir, "vocab"))
        with open(os.path.join(args.package_dir, "package_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "method": "pretrained_masked_prediction_loss",
                "checkpoint_prefix": "ckpt_dir/" + os.path.basename(args.checkpoint),
                "checkpoint_files": copied,
                "config": "zipzap_config.json", "vocab": "vocab",
                "score_direction": "higher masked-token loss means less predictable",
                "trained_on_phishing_labels": False,
            }, f, indent=2)
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(args.seed)
        placeholders, score_tensor, _ = graph_for_scoring(args.config, vocab, args.negative_samples)
        variables = tf.trainable_variables()
        assignment, _ = modeling.get_assignment_map_from_checkpoint(variables, args.checkpoint)
        tf.train.init_from_checkpoint(args.checkpoint, assignment)
        init = tf.global_variables_initializer()
    # Stream windows in bounded batches. The test split can contain many
    # overlapping windows, so retaining every parsed TFRecord in memory can
    # exhaust even a large SageMaker instance.
    aggregate = {}
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=session_config) as session:
        session.run(init)
        for repeat in range(args.mask_repeats):
            batch = []
            processed = 0

            def score_batch(rows):
                nonlocal processed
                if not rows:
                    return
                feed = prepare_batch(
                    rows, args.mask_count,
                    args.seed + repeat * 100003 + processed)
                batch_scores = session.run(score_tensor, feed_dict={
                    placeholders[key]: value for key, value in feed.items()})
                for row, score in zip(rows, batch_scores):
                    entry = aggregate.setdefault(row["address"], {
                        "label": row["label"], "score_sum": 0.0, "count": 0})
                    if entry["label"] != row["label"]:
                        raise ValueError("Inconsistent labels across windows for account %s" % row["address"])
                    entry["score_sum"] += float(score)
                    entry["count"] += 1
                processed += len(rows)

            for row in read_records(args.input_file, args.max_records):
                batch.append(row)
                if len(batch) >= args.batch_size:
                    score_batch(batch)
                    batch = []
            score_batch(batch)
            print("Finished mask repeat %d/%d" % (repeat + 1, args.mask_repeats))

    addresses = sorted(aggregate)
    labels = [aggregate[address]["label"] for address in addresses]
    account_scores = [aggregate[address]["score_sum"] / aggregate[address]["count"]
                      for address in addresses]
    labels = np.asarray(labels, dtype=np.int8)
    account_scores = np.asarray(account_scores, dtype=np.float32)
    if len(np.unique(labels)) < 2:
        roc = ap = None
    else:
        roc = float(roc_auc_score(labels, account_scores))
        ap = float(average_precision_score(labels, account_scores))
    k = max(1, int(np.ceil(0.01 * len(labels))))
    top = np.argsort(-account_scores, kind="mergesort")[:k]
    p_at_k = float(labels[top].mean())
    r_at_k = float(labels[top].sum() / max(1, labels.sum()))
    result = {
        "method": "pretrained_masked_prediction_loss",
        "accounts": len(labels), "known_phishing_labels": int(labels.sum()),
        "roc_auc_observed_labels": roc,
        "average_precision_observed_labels": ap,
        "precision_at_1pct_observed_labels": p_at_k,
        "recall_at_1pct_observed_labels": r_at_k,
        "f1_at_1pct_alert_budget_observed_labels": float(
            2 * p_at_k * r_at_k / max(1e-12, p_at_k + r_at_k)),
        "flagged_accounts_at_1pct_alert_budget": int(k),
        "mask_count_per_window": args.mask_count,
        "mask_repeats": args.mask_repeats,
        "negative_samples": args.negative_samples,
        "seed": args.seed,
        "note": "Account-level means across windows and mask repeats; class 0 is unlabelled.",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["address_id", "observed_label", "masked_prediction_loss"])
        writer.writerows(zip(addresses, labels.tolist(), account_scores.tolist()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
