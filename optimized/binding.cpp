#include <torch/extension.h>
#include "softmaxone.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foreard", &softmax_one_forward, "Softmax1 forward pass");
    m.def("backward", &softmax_one_backward, "Softmax1 backward pass")
}


