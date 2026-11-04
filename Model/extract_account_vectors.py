"""Export one frozen pretrained ZipZap vector per account from a TFRecord split.

Repeated sequence windows for an account are averaged before writing, so each
account contributes exactly one row to downstream PU/BCE head training.
"""
import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import modeling


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="One fixed split TFRecord")
    parser.add_argument("--output_file", required=True, help="Output account-vector NPZ")
    parser.add_argument("--checkpoint", default="./ckpt_dir/model_72402",
                        help="Original pretrained ZipZap checkpoint prefix")
    parser.add_argument("--config", default="./zipzap_config.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_records", type=int, default=0,
                        help="Optional diagnostic cap; 0 exports the whole split")
    return parser.parse_args()


def records(path, max_records):
    iterator = tf.python_io.tf_record_iterator(path)
    for index, serialized in enumerate(iterator):
        if max_records and index >= max_records:
            break
        ex = tf.train.Example.FromString(serialized)
        f = ex.features.feature
        integer = lambda key: np.asarray(f[key].int64_list.value, dtype=np.int32)
        yield {
            "address": int(integer("address")[0]),
            "label": int(round(f["label"].float_list.value[0])),
            "input_ids": integer("input_ids"),
            "input_positions": integer("input_positions"),
            "input_counts": integer("input_counts"),
            "input_mask": integer("input_mask"),
            "input_io_flags": integer("input_io_flags"),
            "input_values": integer("input_values"),
        }


def main():
    args = arguments()
    config = modeling.BertConfig.from_json_file(args.config)
    graph = tf.Graph()
    with graph.as_default():
        inputs = {
            key: tf.placeholder(tf.int32, [None, None], name=key)
            for key in ("input_ids", "input_positions", "input_counts", "input_mask",
                        "input_io_flags", "input_values")
        }
        model = modeling.BertModel(
            config=config, is_training=False,
            input_ids=inputs["input_ids"], input_positions=inputs["input_positions"],
            input_io_flags=inputs["input_io_flags"], input_amounts=inputs["input_values"],
            input_counts=inputs["input_counts"], input_mask=inputs["input_mask"],
            token_type_ids=None, use_one_hot_embeddings=False, cross_share=True)
        account_vectors = model.get_sequence_output()[:, 0, :]
        variables = tf.trainable_variables()
        assignment, _ = modeling.get_assignment_map_from_checkpoint(variables, args.checkpoint)
        tf.train.init_from_checkpoint(args.checkpoint, assignment)
        init = tf.global_variables_initializer()

    sums, counts, labels_by_account = {}, {}, {}
    batch = []
    with tf.Session(graph=graph) as session:
        session.run(init)

        def flush(rows):
            if not rows:
                return
            feed = {inputs[key]: np.stack([row[key] for row in rows])
                    for key in inputs}
            vectors = session.run(account_vectors, feed_dict=feed)
            for row, vector in zip(rows, vectors):
                account = row["address"]
                if account in labels_by_account and labels_by_account[account] != row["label"]:
                    raise ValueError("Inconsistent labels across windows for account %s" % account)
                labels_by_account[account] = row["label"]
                sums[account] = sums.get(account, np.zeros_like(vector, dtype=np.float64)) + vector
                counts[account] = counts.get(account, 0) + 1

        for row in records(args.input_file, args.max_records):
            batch.append(row)
            if len(batch) == args.batch_size:
                flush(batch)
                batch = []
        flush(batch)

    addresses = np.asarray(sorted(sums), dtype=np.int64)
    vectors = np.stack([sums[a] / counts[a] for a in addresses]).astype(np.float32)
    labels = np.asarray([labels_by_account[a] for a in addresses], dtype=np.int8)
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(args.output_file, addresses=addresses, labels=labels, vectors=vectors)
    print("Wrote %d accounts; positive labels=%d; vector width=%d -> %s" % (
        len(addresses), int(labels.sum()), vectors.shape[1], args.output_file))


if __name__ == "__main__":
    main()
