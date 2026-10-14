"""UNSW-NB15 network intrusion detection dataset (Moustafa & Slay, 2015).

Uses the predefined train/test partition (175,341 / 82,332 flows). Two on-disk
formats are accepted inside ``<data_dir>/unsw-nb15/``:
  * the official CSVs  UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv
  * the parquet version of the Kaggle "dhoogla/unswnb15" release (train.parquet /
    test.parquet, downloaded automatically from a Hugging Face mirror if nothing is found).
The larger file is used for training, as in the official release (the parquet mirror names
the two files the other way round).

Preprocessing: categorical columns (proto, service, state) are one-hot encoded with the
categories seen in training; numeric columns go through sign(x) * log1p(|x|) and are
standardised with training statistics.

Tasks: ``multiclass`` = 10 classes of attack_cat (Normal + 9 attacks), ``binary`` = label.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

MIRROR = "https://huggingface.co/datasets/wwydmanski/UNSW-NB15/resolve/main/"
CSV_FILES = ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
PARQUET_FILES = ("train.parquet", "test.parquet")
CATEGORICAL = ("proto", "service", "state")
DROP = ("id", "attack_cat", "label")
ATTACK_CLASSES = ("Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
                  "Analysis", "Backdoor", "Shellcode", "Worms")


def _read_frames(data_dir):
    root = os.path.join(data_dir, "unsw-nb15")
    csvs = [os.path.join(root, f) for f in CSV_FILES]
    pqs = [os.path.join(root, f) for f in PARQUET_FILES]
    if all(os.path.exists(p) for p in csvs):
        frames = [pd.read_csv(p) for p in csvs]
    else:
        if not all(os.path.exists(p) for p in pqs):
            os.makedirs(root, exist_ok=True)
            for f, p in zip(PARQUET_FILES, pqs):
                if not os.path.exists(p):
                    print(f"Downloading {MIRROR + f} -> {p}")
                    urllib.request.urlretrieve(MIRROR + f, p + ".part")
                    os.replace(p + ".part", p)
        try:
            frames = [pd.read_parquet(p) for p in pqs]
        except ImportError as e:
            raise ImportError("reading the UNSW-NB15 parquet files needs pyarrow: "
                              "pip install pyarrow (or place the official CSVs in "
                              f"{root})") from e
    frames.sort(key=len, reverse=True)  # training set = the larger one
    train, test = frames
    for df in (train, test):
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor"})
    common = [c for c in train.columns if c in test.columns]
    return train[common], test[common]


def load_unsw(data_dir, task="multiclass"):
    """Returns (x_train float32 [N, F], y_train int64, x_test, y_test, class_names)."""
    train, test = _read_frames(data_dir)
    if task == "binary":
        classes = ("Normal", "Attack")
        y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    else:
        classes = ATTACK_CLASSES
        lookup = {c: i for i, c in enumerate(classes)}
        y_tr = train["attack_cat"].map(lookup).to_numpy()
        y_te = test["attack_cat"].map(lookup).to_numpy()
        if np.isnan(y_tr.astype(float)).any() or np.isnan(y_te.astype(float)).any():
            raise ValueError("unexpected attack_cat values: "
                             f"{set(train['attack_cat']) - set(classes)}")

    features = [c for c in train.columns if c not in DROP]
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]

    def numeric(df):
        x = df[num].to_numpy(dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    x_num_tr, x_num_te = numeric(train), numeric(test)
    mean, std = x_num_tr.mean(axis=0), x_num_tr.std(axis=0) + 1e-6
    x_num_tr, x_num_te = (x_num_tr - mean) / std, (x_num_te - mean) / std

    onehots_tr, onehots_te = [], []
    for c in cat:
        cats = pd.Categorical(train[c].astype(str)).categories
        onehots_tr.append(pd.get_dummies(pd.Categorical(train[c].astype(str), categories=cats)).to_numpy())
        onehots_te.append(pd.get_dummies(pd.Categorical(test[c].astype(str), categories=cats)).to_numpy())

    x_tr = np.concatenate([x_num_tr] + onehots_tr, axis=1).astype(np.float32)
    x_te = np.concatenate([x_num_te] + onehots_te, axis=1).astype(np.float32)
    return x_tr, y_tr.astype(np.int64), x_te, y_te.astype(np.int64), classes
