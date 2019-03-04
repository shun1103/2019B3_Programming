
# ex15.py
# 2018/02/19 Yuma Saito

import numpy as np

x1 = np.array([1, 0, 0, 1])
x2 = np.array([0, 1, 0, 1])

def cosine_sim(x1, x2):
    # np.dot：二つのベクトルの内積を計算
    # np.linalg.norm：ベクトルのノルムを計算
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

print(cosine_sim(x1, x2))
