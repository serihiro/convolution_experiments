import math

import numpy as np
from chainer.utils.conv import im2col_cpu
from my_convolution.src.my_convolution import im2col


class TestIm2Col:
    def test_basis(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 0
        stride = 1

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)

        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_pad_1(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 1
        stride = 1

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_pad_2(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 2
        stride = 1

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_pad_5(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 5
        stride = 1

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_stride_2(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 0
        stride = 2

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_stride_3(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 0
        stride = 3

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_pad_1_and_stride_2(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 1
        stride = 2

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)
        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    def test_pad_2_and_stride_2(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        h_k = 2
        w_k = 2
        pad = 2
        stride = 2

        expected = im2col_cpu(img=x, kh=h_k, kw=w_k, sy=stride, sx=stride, ph=pad, pw=pad).tolist()
        actual = im2col(x.tolist(), h_k=h_k, w_k=w_k, pad=pad, stride=stride)

        self.__assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride)

    @staticmethod
    def __assert_eq_arrays(actual, expected, n, c_i, h_i, w_i, h_k, w_k, pad, stride):
        h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
        w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

        for nn in range(n):
            for cc in range(c_i):
                for hh_k in range(h_k):
                    for ww_k in range(w_k):
                        for hh in range(h_o):
                            for ww in range(w_o):
                                assert actual[nn][cc][hh_k][ww_k][hh][ww] \
                                       == expected[nn][cc][hh_k][ww_k][hh][ww]  # NOQA
