import math

import numpy as np
import pytest
from chainer.functions.connection.convolution_2d import convolution_2d
from my_convolution.src.my_convolution import convolution_with_im2col


class TestConvolutionWithIm2col:
    def test_basis(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_basis_with_1_1_kernel(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 1
        w_k = 1
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_stride_2(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 2
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_stride_3(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_stride_4(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 4
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_pad_1(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 1

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_stride_3_and_pad_1(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 1

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_stride_3_and_pad_2(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 2

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_n_3(self):
        n = 3
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_n_3_stride_3_and_pad_1(self):
        n = 3
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 1

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_n_3_stride_3_and_pad_2(self):
        n = 3
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 2

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_i_2(self):
        n = 1
        c_i = 2
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_i_3(self):
        n = 1
        c_i = 2
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_i_3_n_4_stride_2_and_pad_2(self):
        n = 4
        c_i = 3
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 2
        pad = 2

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_2(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 2
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_3(self):
        n = 1
        c_i = 1
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 3
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_3_and_c_i_2(self):
        n = 1
        c_i = 2
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 3
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_3_and_c_i_3(self):
        n = 1
        c_i = 3
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 3
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_3_c_i_3_and_n_3(self):
        n = 3
        c_i = 3
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 3
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 1
        pad = 0

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_with_c_o_3_c_i_3_n_3_stride_3_and_padding_5(self):
        n = 3
        c_i = 3
        h_i = 10
        w_i = 10
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 3
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        stride = 3
        pad = 5

        expected = convolution_2d(x, W, stride=stride, pad=pad).data.tolist()
        actual = convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

        self.__assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride)

    def test_assertion_error_with_too_large_kernel(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 4  # > h_i
        w_k = 4  # > w_i
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        with pytest.raises(AssertionError):
            convolution_with_im2col(x.tolist(), W.tolist())

    def test_assertion_error_with_too_large_stride(self):
        n = 1
        c_i = 1
        h_i = 3
        w_i = 3
        x = np.arange(n * c_i * h_i * w_i).reshape(n, c_i, h_i, w_i).astype(np.float32)
        c_o = 1
        h_k = 2
        w_k = 2
        W = np.full(c_o * c_i * h_k * w_k, 2).reshape(c_o, c_i, h_k, w_k).astype(np.float32)

        pad = 1
        stride = 5  # > h_i - h_k + 2 * pad + 1

        with pytest.raises(AssertionError):
            convolution_with_im2col(x.tolist(), W.tolist(), stride=stride, pad=pad)

    @staticmethod
    def __assert_eq_arrays(actual, expected, n, c_o, h_i, w_i, h_k, w_k, pad, stride):
        h_o = math.floor((h_i - h_k + 2 * pad) / float(stride)) + 1
        w_o = math.floor((w_i - w_k + 2 * pad) / float(stride)) + 1

        for nn in range(n):
            for cc in range(c_o):
                for hh in range(h_o):
                    for ww in range(w_o):
                        assert actual[nn][cc][hh][ww] == expected[nn][cc][hh][ww]
