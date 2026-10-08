# LLM Applications in Finance: Ethereum Fraud Detection

This repository contains the replication code for detecting Ethereum fraud using a language model approach, based on the ZipZap paper architecture. 

## Requirements
* Python >= 3.6
* TensorFlow
* tf-keras (Required for legacy Keras layer compatibility)
* NumPy
* Pandas
* Scikit-Learn

```bash
pip install tensorflow tf-keras numpy pandas scikit-learn
```

## How to run the Pre-trained Evaluation Locally

If you have been sent the pre-trained weights (`ckpt_dir`) and test dataset (`data`) separately (they are not tracked in GitHub due to file size limits), follow these steps to evaluate the model on your machine:

### 1. Place the files in the correct directories
Extract the provided files directly into the `Model/` directory so the folder structure looks exactly like this:

```
ZipZap-main/
└── Model/
    ├── ckpt_dir/
    │   ├── bert_finetune.data-00000-of-00001
    │   ├── bert_finetune.index
    │   ├── bert_finetune.meta
    │   └── checkpoint
    ├── data/
    │   ├── finetune_test.tfrecord
    │   └── vocab
    ├── run_finetune.py
    └── ...
```

### 2. Run the evaluation script
Navigate into the `Model` directory and execute the evaluation command. 

> **Important note:** You must include `--load_cross=False` when running the evaluation, otherwise the script will attempt to search for a shared layer that was deactivated during our standard pre-training.

```bash
cd Model
python run_finetune.py --do_train=False --do_eval=True --init_checkpoint=ckpt_dir/bert_finetune --load_cross=False
```

### 3. Understanding the output
The script will loop through the `finetune_test.tfrecord` dataset. Once it reaches the end of the sequence, it will print out the final Area Under the Curve (AUC) metric and classification reports for various thresholds (0.05 to 0.95). 

A standard threshold of `0.5` should yield an AUC of **~0.9766**.
