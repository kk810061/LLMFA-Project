# SageMaker run for PU and masked-prediction experiments

The GitHub clone contains the scripts and config, but Git intentionally excludes the large Kaggle data, pretrained checkpoint, and vocabulary. Before launching, stage two S3 prefixes:

1. `kaggle/`: the extracted Kaggle CSVs directly at the prefix root, including the two `normal_eoa_transaction_*_1000K.csv` files, `phisher_transaction_in.csv`, `phisher_transaction_out.csv`, and `phisher_account.txt`.
2. `zipzap_assets/`: `model_72402.index`, its matching `.data-*` checkpoint shard (and `.meta` if present), and the exact `vocab` file used with that checkpoint. Do not use `bert_finetune` here.

The source dataset shown in Kaggle is private, so either upload its extracted files to an S3 bucket your SageMaker role can read, or arrange Kaggle authentication in SageMaker without committing the Kaggle token to GitHub.

Run this in a SageMaker Studio notebook after cloning the repo and replacing the bucket name:

```python
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

bucket = "YOUR_BUCKET"
estimator = TensorFlow(
    entry_point="sagemaker_train.py",
    source_dir="Model",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.g5.8xlarge",
    framework_version="2.12.0",
    py_version="py310",
    volume_size=200,
    max_run=86400,
    output_path=f"s3://{bucket}/zipzap-training",
    hyperparameters={
        "dataset": "1M",
        "epochs": 100,
        "batch-size": 256,
        "priors": "0.0087,0.015,0.03,0.05",
        "label-fractions": "1.0,0.5,0.25",
        "seeds": "17,29,43",
    },
)
estimator.fit({
    "kaggle": f"s3://{bucket}/zipzap/kaggle/",
    "zipzap_assets": f"s3://{bucket}/zipzap/assets/",
})
print("Download the model artifact from:", estimator.model_data)
```

The 1M Kaggle input in the current project is about 5.8 GB. The 200 GB training volume leaves working room for the sequence dictionary and temporary TFRecords. The scratch TFRecords and account-vector NPZs go under `/tmp`; only reports go under `/opt/ml/output/data/zipzap_research`, and downloadable model files go under `/opt/ml/model`.

The job runs the deterministic 70/15/15 account split, frozen-vector export, nnPU plus same-split BCE, the unsupervised baselines, and masked-loss scoring. It writes `comparison.csv`/`comparison.json` and per-method metrics under the SageMaker output-data directory. SageMaker packages `/opt/ml/model` as the downloadable model artifact and uploads `/opt/ml/output/data` as supplementary output. The model artifact contains the saved Keras heads and the pretrained ZipZap checkpoint/config/vocabulary used by masked prediction.

The full default run is a sweep: three label-availability levels, three seeds, the requested prior grid plus an estimated prior, BCE references, three unsupervised methods, and masked scoring. To get a shorter first result, reduce the hyperparameters in the snippet to one seed, one prior, one label fraction, and fewer epochs. Treat that as a smoke run; use the full settings for the multi-seed/sensitivity result.

For a shorter presentation run, use `epochs=30`, `ae-epochs=20`, `seeds="17"`, `priors="0.03"`, `label-fractions="1.0"`, and `mask-repeats=1`. The estimated prior is still added, so this runs the prior estimate and `0.03` where distinct, plus the BCE reference. This is a faster single-seed result, not the full sensitivity analysis.

For local use, download/unpack the model artifact. The PU `.keras` files score account vectors with `predict_pu_model.py`; use the packaged pretrained checkpoint and vocabulary with `extract_account_vectors.py` to generate those vectors from the TFRecords. The masked-prediction package contains its required pretrained weights and assets, and `score_masked_prediction.py` runs the scorer locally. Metrics remain observed-label metrics because unreported class-0 accounts may include phishers.
