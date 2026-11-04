"""Run a downloaded nnPU/BCE Keras head on account vectors exported by ZipZap."""
import argparse
import csv
import os

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Downloaded .keras model")
    parser.add_argument("--vectors", required=True, help="NPZ with addresses and vectors")
    parser.add_argument("--output", default="pu_predictions.csv")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    with np.load(args.vectors, allow_pickle=False) as data:
        addresses = np.asarray(data["addresses"]).reshape(-1)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or len(addresses) != len(vectors):
        raise ValueError("Expected one vector row per address")
    if int(model.input_shape[-1]) != vectors.shape[1]:
        raise ValueError("Vector width does not match the downloaded model")

    logits = np.asarray(model.predict(vectors, verbose=0)).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["address_id", "phishing_score"])
        writer.writerows(zip(addresses.tolist(), scores.tolist()))
    print("Wrote %d account scores to %s" % (len(scores), args.output))


if __name__ == "__main__":
    main()
