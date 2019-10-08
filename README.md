[![Build Status](https://travis-ci.org/serihiro/convolution_experiments.svg?branch=master)](https://travis-ci.org/serihiro/convolution_experiments)
[![Coverage Status](https://coveralls.io/repos/github/serihiro/convolution_experiments/badge.svg?branch=master)](https://coveralls.io/github/serihiro/convolution_experiments?branch=master)

# About this repository
- This repository contains my some convolution implementations

# Convolution implementations
- All convolution implenentations are implemented in `python/my_convolution/src/my_convolution.py`.

## direct style
- `my_convolution.convolution_with_numpy(
   x: np.ndarray, 
   W: np.ndarray, 
   stride: int = 1, 
   pad: int = 0)`
- `my_convolution.convolution_with_standard_library(
  x: List[List[List[List[float]]]],
  W: List[List[List[List[float]]]],
  stride: int = 1,
  pad: int = 0)`

## im2col style
- `my_convolution.convolution_with_im2col(
  x: List[List[List[List[float]]]], 
  W: List[List[List[List[float]]]],
  stride: int = 1, 
  pad: int = 0)`
- `my_convolution.convolution_with_im2col_and_gemm(
  x: List[List[List[List[float]]]], 
  W: List[List[List[List[float]]]],
  stride: int = 1,
  pad: int = 0)`

# Tests for convolution implementations

```sh
pytest python/my_convolution/tests
```
