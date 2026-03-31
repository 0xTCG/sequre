"""
ChestMNIST Data Preparation
============================
Downloads the ChestMNIST dataset via the ``medmnist`` package and saves
flattened images and multi-label binary labels as whitespace-delimited text
files that the Codon CNN application can load with ``loadtxt``.

ChestMNIST summary
------------------
* 28 × 28 grayscale chest X-ray thumbnails
* 14 binary disease labels per image (multi-label classification)
* Official split — train: 78 468, val: 11 219, test: 22 433

Usage
-----
    cd applications/offline
    pip install medmnist
    python chestmnist_prep.py [--n_train 256] [--n_test 128]

Output (relative to repo root)
------------------------------
    data/chestmnist/train_images.txt   (N_train, 784)  float in [0, 1]
    data/chestmnist/train_labels.txt   (N_train, 14)   int {0, 1}
    data/chestmnist/test_images.txt    (N_test,  784)  float in [0, 1]
    data/chestmnist/test_labels.txt    (N_test,  14)   int {0, 1}
"""

import argparse
import os

import numpy as np
from medmnist import ChestMNIST


def main():
    parser = argparse.ArgumentParser(description="Prepare ChestMNIST data for Sequre CNN")
    parser.add_argument("--n_train", type=int, default=256,
                        help="Number of training samples to export (default: 256)")
    parser.add_argument("--n_test", type=int, default=128,
                        help="Number of test samples to export (default: 128)")
    parser.add_argument("--out_dir", type=str, default="data/chestmnist",
                        help="Output directory relative to repo root (default: data/chestmnist)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Downloading / loading ChestMNIST ...")
    train_ds = ChestMNIST(split="train", download=True, root=args.out_dir)
    test_ds  = ChestMNIST(split="test",  download=True, root=args.out_dir)

    n_train = min(args.n_train, len(train_ds.imgs))
    n_test  = min(args.n_test,  len(test_ds.imgs))

    # Images: uint8 [0, 255] → float64 [0, 1], then flatten to (N, 784)
    train_imgs = train_ds.imgs[:n_train].astype(np.float64) / 255.0
    test_imgs  = test_ds.imgs[:n_test].astype(np.float64)   / 255.0

    train_flat = train_imgs.reshape(n_train, -1)
    test_flat  = test_imgs.reshape(n_test,  -1)

    # Labels: (N, 14) binary {0, 1}
    train_lbls = train_ds.labels[:n_train].astype(np.float64)
    test_lbls  = test_ds.labels[:n_test].astype(np.float64)

    # Save as whitespace-delimited text (one row per sample)
    np.savetxt(os.path.join(args.out_dir, "train_images.txt"), train_flat, fmt="%.6f")
    np.savetxt(os.path.join(args.out_dir, "train_labels.txt"), train_lbls, fmt="%d")
    np.savetxt(os.path.join(args.out_dir, "test_images.txt"),  test_flat,  fmt="%.6f")
    np.savetxt(os.path.join(args.out_dir, "test_labels.txt"),  test_lbls,  fmt="%d")

    print(f"\nSaved to {os.path.abspath(args.out_dir)}/")
    print(f"  train_images.txt  shape=({n_train}, 784)  dtype=float")
    print(f"  train_labels.txt  shape=({n_train}, 14)   dtype=int")
    print(f"  test_images.txt   shape=({n_test}, 784)   dtype=float")
    print(f"  test_labels.txt   shape=({n_test}, 14)    dtype=int")


if __name__ == "__main__":
    main()
