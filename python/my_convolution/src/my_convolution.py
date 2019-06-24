"""My convolution implementations."""

import math

import numpy as np

INITIAL_VALUE = 0.0


def pad_tensor(x: list, n: int, c_i: int, h_i: int, w_i: int, pad: int) -> None:
    """Pad elements to the tensor.

    Args:
        x (list):
        n (int):
        c_i (int):
        h_i (int):
        w_i (int):
        pad (int):

    Returns:
        None

    """
    for nn in range(n):
        for cc_i in range(c_i):

            for pp in range(pad):
                x[nn][cc_i].insert(pp,
                                   [INITIAL_VALUE for _i in range(w_i + 2 * pad)])
                x[nn][cc_i].insert(h_i + pp + 1,
                                   [INITIAL_VALUE for _i in range(w_i + 2 * pad)])

            for hh_i in range(1, h_i + pad):
                for pp in range(pad):
                    x[nn][cc_i][hh_i].insert(pp, INITIAL_VALUE)
                    x[nn][cc_i][hh_i].insert(w_i + pp + 1, INITIAL_VALUE)


def im2col(x: list, h_k: int, w_k: int, pad: int = 0, stride: int = 1) -> list:
    """Convert the tensor to a vector.

    Args:
        x (list):
        h_k (int):
        w_k (int):
        pad (int):
        stride (int):

    Returns:
        list

    """
    n = len(x)
    c_i = len(x[0])
    h_i = len(x[0][0])
    w_i = len(x[0][0][0])
    h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
    w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

    if pad > 0:
        pad_tensor(x, n, c_i, h_i, w_i, pad)

    result = [[[[[[INITIAL_VALUE for _i in range(w_o)]
                  for _j in range(h_o)]
                 for _k in range(h_k)]
                for _l in range(w_k)]
               for _m in range(c_i)]
              for _n in range(n)]

    for nn in range(n):
        for cc_i in range(c_i):
            for hh_o in range(h_o):
                for ww_o in range(w_o):
                    for hh_k in range(h_k):
                        for ww_k in range(w_k):
                            result[nn][cc_i][hh_k][ww_k][hh_o][ww_o] = \
                                x[nn][cc_i][hh_o * stride + hh_k][ww_o * stride + ww_k]

    return result


def convolution_with_numpy(x: np.ndarray, W: np.ndarray, stride: int = 1, pad: int = 0) \
        -> np.ndarray:
    r"""Convolution implementation using Numpy.

    Args:
        x (numpy.ndarray): Input image whose shape consists of ``n``, ``c_i``, ``h``, and ``w``,
                         where ``n`` is the size of batch, ``c_i`` is the size of input channel,
                         ``h`` is the size of height of the image,
                         and ``w`` is the size of width of the image.
        W (numpy.ndarray): Kernel whose shape consists of ``c_o``, ``c_i``, ``h``, and ``w``,
                         where ``c_o`` is the size of output channel, ``c_i`` is the size of
                         input channel, ``h`` is the size of height of the kernel,
                         and ``w`` is the size of width of the kernel.

        stride (int): stride size. The default value is ``1``.
        pad (int): padding size. The default value is  ``0``.

    Returns:
        numpy.nadarray: ndarray object whose shape consists of ``n``, ``c_o``,
        ``h_o``, ``w_o``, where ``h_o`` is
        the result of ``math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1``,
        and `w_o` is the result of ``math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1``.

    Raises:
        AssertionError: If ``h_k > h_i or w_k > w_i`` or \
        ``stride > (h_i - h_k + 2 * pad + 1) or stride > (w_i - w_k + 2 * pad + 1)``

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


def convolution_with_standard_library(x: list, W: list, stride: int = 1, pad: int = 0) \
        -> list:
    r"""Convolution implementation using only Python standard library.

    Args:
        x (list): Input image whose shape consists of ``n``, ``c_i``, ``h``, and ``w``,
                         where ``n`` is the size of batch, ``c_i`` is the size of input channel,
                         ``h`` is the size of height of the image,
                         and ``w`` is the size of width of the image.
        W (list): Kernel whose shape consists of ``c_o``, ``c_i``, ``h``, and ``w``,
                         where ``c_o`` is the size of output channel, ``c_i`` is the size of
                         input channel, ``h`` is the size of height of the kernel,
                         and ``w`` is the size of width of the kernel.

        stride (int): stride size. The default value is ``1``.
        pad (int): padding size. The default value is  ``0``.

    Returns:
        list: list object whose shape consists of ``n``, ``c_o``,
        ``h_o``, ``w_o``, where ``h_o`` is
        the result of ``math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1``,
        and `w_o` is the result of ``math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1``.

    Raises:
        AssertionError: If ``h_k > h_i or w_k > w_i`` or \
        ``stride > (h_i - h_k + 2 * pad + 1) or stride > (w_i - w_k + 2 * pad + 1)``

    """
    n = len(x)
    c_i = len(x[0])
    h_i = len(x[0][0])
    w_i = len(x[0][0][0])

    c_o = len(W)
    c_i = len(W[0])
    h_k = len(W[0][0])
    w_k = len(W[0][0][0])

    if h_k > h_i or w_k > w_i:
        raise AssertionError('The height and width of x must be smaller than W')

    if stride > (h_i - h_k + 2 * pad + 1) or stride > (w_i - w_k + 2 * pad + 1):
        raise AssertionError('The value of stride must be smaller than output tensor size')

    h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
    w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

    if pad > 0:
        pad_tensor(x, n, c_i, h_i, w_i, pad)

    result = [[[[INITIAL_VALUE for _i in range(w_o)]
                for _j in range(h_o)]
               for _k in range(c_o)]
              for _h in range(n)]

    for nn in range(n):
        for cc_i in range(c_i):
            for cc_o in range(c_o):
                for h in range(h_o):
                    for w in range(w_o):
                        for k_h in range(h_k):
                            for k_w in range(w_k):
                                result[nn][cc_o][h][w] += \
                                    x[nn][cc_i][h * stride + k_h][w * stride + k_w] * \
                                    W[cc_o][cc_i][k_h][k_w]

    return result
