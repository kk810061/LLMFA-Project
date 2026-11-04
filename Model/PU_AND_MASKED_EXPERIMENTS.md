# PU and masked-prediction experiments

These experiments are separate from `run_finetune.py`; the existing supervised model and its checkpoints remain intact.

For a single SageMaker job that runs the full pipeline and writes downloadable model and result artifacts, follow [`SAGEMAKER_QUICKSTART.md`](SAGEMAKER_QUICKSTART.md) and use `sagemaker_train.py` as the entry point.

## Methods

- `extract_account_vectors.py` runs the frozen pretrained ZipZap encoder over each account's windows and averages the `[0]` account-token vector. It writes one vector per account.
- `run_pu_learning.py` trains the current two-layer residual MLP head with either nnPU risk or BCE. nnPU uses known positives as P and every remaining training account as U. It never treats training label 0 as a verified negative. The BCE run is the known-label baseline on the same vectors.
- `run_unsupervised_baselines.py` reruns Isolation Forest, K-means, and an autoencoder on those same split vectors. It fits without labels and uses the same validation/test metric protocol.
- The class-prior diagnostic is the cross-fitted Elkan-Noto label-propensity ratio, computed from training vectors and labels only. It assumes positives are selected at random for labelling. The observed positive rate is a lower bound; the default sensitivity values and the estimate are all run.
- `score_masked_prediction.py` scores test transactions with the original pretrained ZipZap MLM checkpoint. It masks transaction tokens (not the focal account ID), averages sampled-token loss over windows and repeated masks, and does not train on phishing labels.

Masked loss is not frequency-normalized in this first version. ZipZap's address-frequency embeddings and negative-sampling distribution can affect the score, so interpret it as an initial test of the prediction-loss hypothesis; frequency calibration is a follow-up analysis.

All reported precision/recall-style metrics are against **observed phishing labels** because class 0 is unlabelled. Rankings include ROC-AUC, Average Precision, Precision@1%, and Recall@1%. PU thresholds are chosen from validation labels only. Masked prediction is a ranking score and has no learned decision threshold.

## Required split artifacts

`prepare_research_splits.py` builds the inputs from the raw Kaggle files already used by this project. It calls the existing ZipZap sequence builder, creates a reproducible stratified 70/15/15 account split (seed 12345 by default), and writes one TFRecord and a split manifest for each partition. It splits accounts before generating overlapping windows, so an account cannot appear in multiple partitions.

The existing generator in this checkout only makes a 70/30 train/test split and has no saved split manifest. The new 70/15/15 split is therefore a reproducible research split; rerun the unsupervised baselines on this split before comparing their scores with the new models. The pretrained checkpoint and its exact matching `data/vocab` must also be staged in SageMaker; they are ignored by Git because of their size.

## Dataset location

`prepare_research_splits.py` accepts the directory containing the extracted Kaggle CSVs. In this checkout, that directory is the repository's `Data/` folder, so a local run can use `--raw_data_dir ../Data` from `Model/`. SageMaker cannot read a folder on your Windows computer directly: upload the extracted contents of `Data/` to S3 and attach that S3 prefix as a training input (or download the private Kaggle dataset into the SageMaker job). Then point `--raw_data_dir` to the mounted input directory, for example `/opt/ml/input/data/kaggle`. The names in the current `Data/` folder match the script's expected Kaggle filenames. The earlier 1M folder here contains about 5.8 GB of CSVs, so allow for that upload and storage size.

## Commands

Run from `Model/` after the fixed split TFRecords are available:

```bash
python prepare_research_splits.py --raw_data_dir /opt/ml/input/data/kaggle --vocab ./data/vocab --output_dir ./research_data/splits
python extract_account_vectors.py --input_file ./research_data/splits/finetune_train.tfrecord --output_file ./research_data/train.npz
python extract_account_vectors.py --input_file ./research_data/splits/finetune_validation.tfrecord --output_file ./research_data/validation.npz
python extract_account_vectors.py --input_file ./research_data/splits/finetune_test.tfrecord --output_file ./research_data/test.npz
python run_pu_learning.py --train ./research_data/train.npz --validation ./research_data/validation.npz --test ./research_data/test.npz --output_dir ./research_results/pu --model_dir /opt/ml/model
python run_unsupervised_baselines.py --train ./research_data/train.npz --validation ./research_data/validation.npz --test ./research_data/test.npz --output_dir ./research_results/unsupervised
python score_masked_prediction.py --input_file ./research_data/splits/finetune_test.tfrecord --output_dir ./research_results/masked_prediction
```

For local preparation with the files already in this repository, replace `/opt/ml/input/data/kaggle` with `../Data`.

When run as a SageMaker training job, the PU script writes a loadable `.keras` file and metadata for every seed/prior/label-fraction run into `/opt/ml/model`. SageMaker packages that directory as the job's model artifact in S3; download the artifact and unpack it to get the weights. Locally, `predict_pu_model.py` applies a selected downloaded `.keras` head to an account-vector NPZ:

```bash
python predict_pu_model.py --model ./downloaded_model/nnpu_pi0p03_labels1p0_seed17.keras --vectors ./research_data/test.npz --output ./pu_predictions.csv
```

The exported nnPU head consumes account vectors, so local transaction-to-score use also needs the base ZipZap encoder checkpoint and the matching vocabulary to run `extract_account_vectors.py`. The project checkpoint `ckpt_dir/model_72402` and `data/vocab` are the required encoder assets.

Masked prediction does not train a new classifier; it scores with the pretrained ZipZap MLM weights. When run in SageMaker, `score_masked_prediction.py` copies those checkpoint files, `zipzap_config.json`, and the vocabulary into `/opt/ml/model` (or the `SM_MODEL_DIR` value), so the pretrained weights are included in the downloadable SageMaker artifact. The scoring script and dependencies are still needed locally to run inference.

For the prescribed label-scarcity analysis, `run_pu_learning.py` defaults to retaining 100%, 50%, and 25% of training positives across three seeds. It re-estimates the prior using only the positives still labelled in each scarcity run, then adds that estimate to the requested sensitivity grid. Change `--priors` and `--label_fractions` to expand the grid. Outputs include metrics JSON/CSV, fixed-threshold reports, per-account predictions, and loadable Keras models when `--model_dir` is provided. The SageMaker entry point writes results and downloadable models to the SageMaker managed output paths.

## Pretrained checkpoint

The vector and masked-prediction commands default to `ckpt_dir/model_72402`, the original pretrained ZipZap checkpoint. The fine-tuned checkpoint `ckpt_dir/bert_finetune` is intentionally not used for the new fixed-vector head because that encoder has already seen phishing labels.
