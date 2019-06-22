import numpy as np
from chainer.functions.connection.convolution_2d import convolution_2d
from my_convolution.src.my_convolution import convolution_with_numpy


class TestConvolusionWithNumpy(object):
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

        expected = convolution_2d(x, W).data
        actual = convolution_with_numpy(x, W)

        assert np.all(np.equal(expected, actual)) == True  # NOQA
