
# load_mnist.py
# MNISTデータセットのダウンロード
# 2018/02/21 Yuma Saito
#
# ------- 使い方 -------
# from dataset.mnist import load_mnist
# mnist = load_mnist()
#
# このmnistデータはnp.int8型のディクショナリです
# mnist["train_img"] -> 訓練画像
# mnist["train_label"] -> 訓練ラベル
# mnist["test_img"] -> テスト画像
# mnist["test_label"] -> テストラベル

import urllib.request
import gzip
import os
import numpy as np

url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = {
    "train_img" : "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img"  : "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))

def download_mnist(filename):
    """
    MNISTデータセットのダウンロード
    filename: ダウンロードするファイル名
    """
    file_path = os.path.join(dataset_dir, filename)
    # そもそもここは呼ばれないはずだが…
    if os.path.exists(file_path):
        return print('already exist')
    print('Downloading ' + filename + ' ...')
    urllib.request.urlretrieve(url_base + filename, file_path)
    print("Done")

def load_mnist(onehot=True, normalize=True, reshape=True):
    """
    MNISTデータセットの読み込み
    ARGS: 読み込み方法の指定。すべてデフォルトはTrue
        onehot:    ラベルをone-hotベクトルにするかどうか
        normalize: 画像データを0.0 ~ 1.0に正規化するかどうか
        reshape: 　画像データを二次元(28x28)にするかどうか
    RETURNS: mnistデータセットのディクショナリ（各valueはnumpy配列）
        mnist["train_img"]: 訓練画像リスト[60000, 28, 28]
        mnist["train_label"]: 訓練ラベルリスト[60000, 10]
        mnist["test_img"]: テスト画像リスト[10000, 28, 28]
        mnist["test_label"]: テストラベルリスト[10000, 10]
    """
    mnist = {}
    for k, v in key_file.items():
        # なかったらダウンロード
        file_path = os.path.join(dataset_dir, v)
        if not os.path.exists(file_path):
            download_mnist(v)
        # ファイルを開く
        with gzip.open(file_path, "rb") as f:
            # 画像データファイルは先頭16Byte分がヘッダ、
            # ラベルデータファイルは先頭8Byte分がヘッダなので
            # その分だけ飛ばして読み込む
            if k == "train_img" or k == "test_img":
                mnist[k] = np.frombuffer(f.read(), np.uint8, offset=16)
            else:
                mnist[k] = np.frombuffer(f.read(), np.uint8, offset=8)

    if onehot:
        mask = np.arange(10)
        for k in ["train_label", "test_label"]:
            mnist[k] = (mnist[k].reshape([1, -1]).T == mask).astype(float)
    if normalize:
        mnist["train_img"] = (mnist["train_img"] / 255.0).astype(float)
        mnist["test_img"] = (mnist["test_img"] / 255.0).astype(float)
    if reshape:
        mnist["train_img"] = mnist["train_img"].reshape([-1, 784])
        mnist["test_img"] = mnist["test_img"].reshape([-1, 784])

    return mnist


def main():
    """
    load_mnistのメイン
    読み込みがうまくできてるどうかのテスト用
    """
    mnist = load_mnist()
    print(mnist["test_label"])


if __name__ == "__main__":
    main()
