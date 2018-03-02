
# ex15.py
# 2018/02/19 Yuma Saito

import numpy as np

def cosine_sim(x1, x2):
    # np.dot：二つのベクトルの内積を計算
    # np.linalg.norm：ベクトルのノルムを計算
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
