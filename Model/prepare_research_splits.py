"""Build fixed account-disjoint train/validation/test TFRecords from Kaggle CSVs.

The raw CSVs are converted with the repository's existing ZipZap sequence
builder. Accounts are split before overlapping sequence windows are created.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MODEL_DIR = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_dir", required=True,
                        help="Directory containing the Kaggle transaction CSVs and phisher_account.txt")
    parser.add_argument("--vocab", required=True,
                        help="ZipZap vocab matching the pretrained checkpoint")
    parser.add_argument("--output_dir", default="./research_data")
    parser.add_argument("--sequences_file", default="",
                        help="Optional existing eoa2seq.pkl; otherwise run gen_seq.py")
    parser.add_argument("--dataset", default="1M", choices=("1M", "1000K", "3M", "3000K", "10M"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--max_windows_per_account", type=int, default=500)
    return parser.parse_args()


def ensure_sequences(args, output_dir):
    if args.sequences_file:
        path = Path(args.sequences_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    raw_dir = Path(args.raw_data_dir).resolve()
    required = [
        raw_dir / "normal_eoa_transaction_in_slice_1000K.csv",
        raw_dir / "normal_eoa_transaction_out_slice_1000K.csv",
        raw_dir / "phisher_transaction_in.csv",
        raw_dir / "phisher_transaction_out.csv",
        raw_dir / "phisher_account.txt",
    ]
    if args.dataset in ("3M", "3000K"):
        required[0] = raw_dir / "normal_eoa_transaction_in_slice_3000K.csv"
        required[1] = raw_dir / "normal_eoa_transaction_out_slice_3000K.csv"
    elif args.dataset == "10M":
        required[0] = raw_dir / "normal_eoa_transaction_in_slice.csv"
        required[1] = raw_dir / "normal_eoa_transaction_out_slice.csv"
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Kaggle dataset files: " + ", ".join(missing))

    sequence_work_dir = output_dir / "sequence_build"
    sequence_work_dir.mkdir(parents=True, exist_ok=True)
    sequence_data_dir = sequence_work_dir / "data"
    sequence_path = sequence_data_dir / "eoa2seq.pkl"
    command = [
        sys.executable, str(MODEL_DIR / "gen_seq.py"),
        "--dataset=" + args.dataset,
        "--data_dir=" + str(raw_dir) + os.sep,
        "--phisher=True", "--dup=True",
    ]
    print("Building ZipZap account sequences from raw Kaggle CSVs...")
    subprocess.run(command, cwd=str(sequence_work_dir), check=True)
    if not sequence_path.is_file():
        raise RuntimeError("Sequence builder did not produce " + str(sequence_path))
    return sequence_path


def make_windows(address, transactions, max_seq_length, max_windows, rng):
    max_transaction_tokens = max_seq_length - 1
    if max_transaction_tokens <= 0:
        raise ValueError("max_seq_length must be at least 2")
    step = max(1, round(max_seq_length * 0.6))
    if len(transactions) <= max_transaction_tokens:
        starts = [0]
    else:
        starts = list(range(len(transactions) - max_transaction_tokens, 0, -step))
        starts.append(0)
        starts = list(reversed(starts))
        if len(starts) > max_windows:
            starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())
    return [[[address, 0, 0, 0, 0, 0]] +
            transactions[start:start + max_transaction_tokens] for start in starts]


def make_splits(accounts, labels, train_fraction, validation_fraction, seed):
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive")
    train_accounts, held_out = train_test_split(
        accounts, test_size=(1.0 - train_fraction), random_state=seed,
        stratify=[labels[a] for a in accounts])
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    validation_accounts, test_accounts = train_test_split(
        held_out, test_size=(1.0 - validation_share), random_state=seed,
        stratify=[labels[a] for a in held_out])
    return {
        "train": sorted(train_accounts),
        "validation": sorted(validation_accounts),
        "test": sorted(test_accounts),
    }


def main():
    args = arguments()
    raw_dir = Path(args.raw_data_dir).resolve()
    vocab_path = Path(args.vocab).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not vocab_path.is_file():
        raise FileNotFoundError("The pretrained model's matching vocab is required: " + str(vocab_path))

    sequence_path = ensure_sequences(args, output_dir)
    with open(sequence_path, "rb") as handle:
        account_sequences = pickle.load(handle)
    phisher_path = raw_dir / "phisher_account.txt"
    phishers = set(pd.read_csv(phisher_path, header=None, dtype=str).iloc[:, 0].str.strip())
    labels = {address: int(str(address).strip() in phishers)
              for address in account_sequences}
    accounts = sorted(account_sequences)
    splits = make_splits(accounts, labels, args.train_fraction,
                         args.validation_fraction, args.seed)
    split_sets = {name: set(ids) for name, ids in splits.items()}
    if (split_sets["train"] & split_sets["validation"] or
            split_sets["train"] & split_sets["test"] or
            split_sets["validation"] & split_sets["test"]):
        raise RuntimeError("Account split overlap detected")

    # Import after consuming our CLI: gen_finetune_data defines TF1 global flags.
    sys.argv[:] = [sys.argv[0]]
    sys.path.insert(0, str(MODEL_DIR))
    import gen_finetune_data as finetune_data
    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    rng = np.random.default_rng(args.seed)
    manifest = {
        "seed": args.seed,
        "fractions": {"train": args.train_fraction,
                      "validation": args.validation_fraction,
                      "test": 1.0 - args.train_fraction - args.validation_fraction},
        "source_sequences": str(sequence_path),
        "vocab": str(vocab_path),
        "splits": {},
    }
    for name, account_ids in splits.items():
        def split_instances():
            for address in account_ids:
                windows = make_windows(address, account_sequences[address],
                                       args.max_seq_length,
                                       args.max_windows_per_account, rng)
                label = float(labels[address])
                for window in windows:
                    yield finetune_data.FinetuneInstance(address, window, label)

        output_file = output_dir / ("finetune_%s.tfrecord" % name)
        records_written = finetune_data.write_finetune_instance_to_example_files(
            split_instances(), args.max_seq_length, vocab, [str(output_file)])
        manifest["splits"][name] = {
            "account_ids": [str(account) for account in account_ids],
            "accounts": len(account_ids),
            "known_phishing_accounts": int(sum(labels[a] for a in account_ids)),
            "records": records_written,
            "tfrecord": output_file.name,
        }
        print("%s: %d accounts, %d known positives, %d windows" % (
            name, len(account_ids), manifest["splits"][name]["known_phishing_accounts"],
            records_written))

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved fixed account split manifest to " + str(manifest_path))


if __name__ == "__main__":
    main()
