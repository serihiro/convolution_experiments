"""My convolution implementations."""

import math

import numpy as np


def convolution_with_numpy(x: np.ndarray, W: np.ndarray, stride: int = 1, pad: int = 0) \
        -> np.ndarray:
    """Convolution implementation using Numpy.

    Args:
        x (numpy.ndarray): Input image whose shape consists of `n`, `c_i`, `h`, and `w`
                         where `n` is the size of batch, `c_i` is the size of input channel,
                         `h` is the size of height of the image,
                         and `w` is the size of width of the image.
        W (numpy.ndarray): Kernel whose shape consists of `c_o`, `c_i`, `h`, and `w`
                         where `c_o` is the size of output channel, `c_i` is the size of
                         input channel, `h` is the size of height of the kernel,
                         and `w` is the size of width of the kernel.

        stride (int): stride size. The default value is `1`.
        pad (int): padding size. The default value is  `0`.

    Returns:
        numpy.nadarray: ndarray object whose shape consists of `n`, `c_o`,
        `h_o`, `w_o`, where `h_o` is
        the result of `math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1`,
        and `w_o` is the result of `math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1`.

    Raises:
        AssertionError: If

    """
    n, c_i, h_i, w_i = x.shape
    c_o, c_i, h_k, w_k = W.shape

    if h_k > h_i or w_k > w_i:
        raise AssertionError('The height and width of x must be smaller than W')

    if stride > (h_i - h_k + 2 * pad + 1) or stride > (w_i - w_k + 2 * pad + 1):
        raise AssertionError('The value of stride must be smaller than output tensor size')

    h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
    w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

    if pad > 0:
        new_x = np.zeros((n, c_i, h_i + 2 * pad, w_i + 2 * pad), dtype=np.float32)
        for nn in range(n):
            for cc_i in range(c_i):
                new_x[nn][cc_i] = np.pad(x[nn][cc_i], pad, 'constant')
        x = new_x

    result = np.zeros((n, c_o, h_o, w_o), dtype=np.float32)

    for nn in range(n):
        for cc_i in range(c_i):
            for cc_o in range(c_o):
                for h in range(h_o):
                    for w in range(w_o):
                        for k_h in range(h_k):
                            for k_w in range(w_k):
                                result[nn, cc_o, h, w] += \
                                    x[nn][cc_i][h * stride + k_h][w * stride + k_w] * \
                                    W[cc_o][cc_i][k_h][k_w]

    return result
