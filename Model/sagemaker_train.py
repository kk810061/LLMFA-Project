"""SageMaker Training entry point for the PU and masked-prediction studies.

Configure two SageMaker input channels:
  kaggle: extracted Kaggle CSVs at the channel root
  zipzap_assets: model_72402.* checkpoint files and the matching vocab file

SageMaker uploads /opt/ml/model as the model artifact and /opt/ml/output/data
as supplementary outputs when the job completes.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", default=os.environ.get("SM_CHANNEL_KAGGLE"))
    parser.add_argument("--asset-dir", default=os.environ.get("SM_CHANNEL_ZIPZAP_ASSETS"))
    parser.add_argument("--dataset", default="1M",
                        choices=("1M", "1000K", "3M", "3000K", "10M"))
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--priors", default="0.0087,0.015,0.03,0.05")
    parser.add_argument("--label-fractions", default="1.0,0.5,0.25")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--mask-repeats", type=int, default=3)
    parser.add_argument("--mask-count", type=int, default=8)
    parser.add_argument("--negative-samples", type=int, default=5000)
    parser.add_argument("--work-dir", default="/tmp/zipzap_research",
                        help="Scratch directory for large TFRecords and vectors")
    parser.add_argument("--output-dir", default=os.path.join(
        os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"), "zipzap_research"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return parser.parse_args()


def locate_asset(asset_dir, pattern):
    matches = sorted(Path(asset_dir).rglob(pattern))
    if not matches:
        raise FileNotFoundError("Could not find %s under %s" % (pattern, asset_dir))
    if len(matches) > 1:
        raise ValueError("Found multiple %s files under %s: %s" % (
            pattern, asset_dir, ", ".join(map(str, matches))))
    return matches[0]


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(MODEL_DIR), check=True)


def build_comparison(report_dir):
    rows = []

    def ranking_f1(precision, recall):
        return (2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0)

    pu_path = report_dir / "pu" / "metrics.json"
    if pu_path.is_file():
        with pu_path.open(encoding="utf-8") as handle:
            pu = json.load(handle)
        for result in pu["results"]:
            rows.append({
                "method": result["method"], "prior": result["prior"],
                "labelled_positive_fraction": result["labelled_positive_fraction"],
                "seed": result["seed"],
                "average_precision_observed_labels": result["average_precision_observed_labels"],
                "roc_auc_observed_labels": result["roc_auc_observed_labels"],
                "precision_at_1pct_observed_labels": result["precision_at_1pct_observed_labels"],
                "recall_at_1pct_observed_labels": result["recall_at_1pct_observed_labels"],
                "f1_at_1pct_alert_budget_observed_labels": ranking_f1(
                    result["precision_at_1pct_observed_labels"],
                    result["recall_at_1pct_observed_labels"]),
                "validation_threshold": result["threshold_from_validation"],
                "precision_at_validation_threshold_observed_labels": result["precision_observed_labels"],
                "recall_at_validation_threshold_observed_labels": result["recall_observed_labels"],
                "f1_at_validation_threshold_observed_labels": result["f1_observed_labels"],
            })

    unsup_path = report_dir / "unsupervised" / "metrics.json"
    if unsup_path.is_file():
        with unsup_path.open(encoding="utf-8") as handle:
            for result in json.load(handle):
                rows.append({
                    "method": result["method"], "prior": None,
                    "labelled_positive_fraction": None, "seed": result["seed"],
                    "average_precision_observed_labels": result["average_precision_observed_labels"],
                    "roc_auc_observed_labels": result["roc_auc_observed_labels"],
                    "precision_at_1pct_observed_labels": result["precision_at_1pct_observed_labels"],
                    "recall_at_1pct_observed_labels": result["recall_at_1pct_observed_labels"],
                    "f1_at_1pct_alert_budget_observed_labels": ranking_f1(
                        result["precision_at_1pct_observed_labels"],
                        result["recall_at_1pct_observed_labels"]),
                    "validation_threshold": result["threshold_from_validation"],
                    "precision_at_validation_threshold_observed_labels": result["precision_observed_labels"],
                    "recall_at_validation_threshold_observed_labels": result["recall_observed_labels"],
                    "f1_at_validation_threshold_observed_labels": result["f1_observed_labels"],
                })

    masked_path = report_dir / "masked_prediction" / "metrics.json"
    if masked_path.is_file():
        with masked_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append({
            "method": result["method"], "prior": None,
            "labelled_positive_fraction": None, "seed": result["seed"],
            "average_precision_observed_labels": result["average_precision_observed_labels"],
            "roc_auc_observed_labels": result["roc_auc_observed_labels"],
            "precision_at_1pct_observed_labels": result["precision_at_1pct_observed_labels"],
            "recall_at_1pct_observed_labels": result["recall_at_1pct_observed_labels"],
            "f1_at_1pct_alert_budget_observed_labels": result["f1_at_1pct_alert_budget_observed_labels"],
            "validation_threshold": None,
            "precision_at_validation_threshold_observed_labels": None,
            "recall_at_validation_threshold_observed_labels": None,
            "f1_at_validation_threshold_observed_labels": None,
        })

    if rows:
        with (report_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (report_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    args = arguments()
    if not args.raw_data_dir or not Path(args.raw_data_dir).is_dir():
        raise FileNotFoundError("Provide --raw-data-dir or a SageMaker 'kaggle' input channel")
    if not args.asset_dir or not Path(args.asset_dir).is_dir():
        raise FileNotFoundError("Provide --asset-dir or a SageMaker 'zipzap_assets' input channel")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")

    checkpoint_index = locate_asset(args.asset_dir, "model_72402.index")
    checkpoint = str(checkpoint_index)[:-len(".index")]
    vocab = locate_asset(args.asset_dir, "vocab")
    work_dir = Path(args.work_dir).resolve()
    report_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    splits_dir = work_dir / "splits"
    vectors_dir = work_dir / "vectors"
    for directory in (splits_dir, vectors_dir, report_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "prepare_research_splits.py",
         "--raw_data_dir", args.raw_data_dir, "--vocab", vocab,
         "--output_dir", splits_dir, "--dataset", args.dataset,
         "--seed", args.split_seed])
    shutil.copy2(splits_dir / "split_manifest.json", report_dir / "split_manifest.json")

    for split in ("train", "validation", "test"):
        run([sys.executable, "extract_account_vectors.py",
             "--input_file", splits_dir / ("finetune_%s.tfrecord" % split),
             "--output_file", vectors_dir / (split + ".npz"),
             "--checkpoint", checkpoint,
             "--config", MODEL_DIR / "zipzap_config.json"])

    run([sys.executable, "run_pu_learning.py",
         "--train", vectors_dir / "train.npz",
         "--validation", vectors_dir / "validation.npz",
         "--test", vectors_dir / "test.npz",
         "--output_dir", report_dir / "pu",
         "--model_dir", model_dir / "pu",
         "--priors", args.priors,
         "--label_fractions", args.label_fractions,
         "--seeds", args.seeds,
         "--epochs", args.epochs,
         "--batch_size", args.batch_size,
         "--learning_rate", args.learning_rate])

    run([sys.executable, "run_unsupervised_baselines.py",
         "--train", vectors_dir / "train.npz",
         "--validation", vectors_dir / "validation.npz",
         "--test", vectors_dir / "test.npz",
         "--output_dir", report_dir / "unsupervised",
         "--seeds", args.seeds,
         "--ae_epochs", args.ae_epochs,
         "--n_estimators", args.n_estimators,
         "--n_clusters", args.n_clusters])

    run([sys.executable, "score_masked_prediction.py",
         "--input_file", splits_dir / "finetune_test.tfrecord",
         "--checkpoint", checkpoint,
         "--config", MODEL_DIR / "zipzap_config.json",
         "--vocab", vocab,
         "--output_dir", report_dir / "masked_prediction",
         "--package_dir", model_dir / "masked_prediction",
         "--mask_repeats", args.mask_repeats,
         "--mask_count", args.mask_count,
         "--negative_samples", args.negative_samples])

    build_comparison(report_dir)
    print("Research outputs: %s" % report_dir, flush=True)
    print("Downloadable model artifacts: %s" % model_dir, flush=True)


if __name__ == "__main__":
    main()
