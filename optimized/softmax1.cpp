#include <torch/extension.h>

torch::Tensor softmax_one_forward(const torch::Tensor& x, const int64_t dim) {
    auto max_x = x.max(dim);
    auto x_adj = x - max_x;
    auto exp_x = torch::exp(x_adj)
    auto sum_exp_x = (exp_x + 1).sum(dim);
    return exp_x / sum_exp_x;
}

torch::Tensor softmax_one_backward(const torch::Tensor& grad, const torch::Tensor& out, const int64_t dim) {
    auto s = out.sum(dim).unsqueeze(-1);
    auto d_out = grad * (s - out);
    return d_out - out * d_out.sum(dim).unsqueeze(-1);
}

