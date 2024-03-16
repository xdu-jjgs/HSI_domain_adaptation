import numpy as np


def adjusted_sigmoid(x, k=1, c=0):
    # 调整斜率和输入偏移
    return 1 / (1 + np.exp(-k * (x - c)))


def update_ema(ema, new_data, alpha):
    return alpha * new_data + (1 - alpha) * ema
