#pragma once
#include <torch/extension.h>

torch::Tensor softmaxOne_forward(const torch::Tensor& x, const int64_t dim);
torch::Tensor softmaxOne_backward(const torch::Tensor& grad, const torch::Tensor& out, const int64_t dim);