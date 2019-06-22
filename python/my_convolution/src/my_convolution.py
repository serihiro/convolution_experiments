import math

import numpy as np


# x: NCHW
# W: C_oC_iHW
def convolution_with_numpy(x: np.ndarray, W: np.ndarray, stride: int = 1, pad: int = 0):
    n, c_i, h_i, w_i = x.shape
    c_o, c_i, h_k, w_k = W.shape

    h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
    w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

    result = np.zeros((n, c_o, h_o, w_o), dtype=np.float32)

    for nn in range(n):
        for cc_i in range(c_i):
            for cc_o in range(c_o):
                for h in range(h_o):
                    for w in range(w_o):
                        for k_h in range(h_k):
                            for k_w in range(w_k):

                                result[nn, cc_i * c_i + cc_o, h, w] += \
                                    x[nn][cc_i][h * stride + k_h][w * stride + k_w] \
                                    * W[nn][cc_o][k_h][k_w]

    return result
