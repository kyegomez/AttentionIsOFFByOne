#include <torch/extension.h>
#include "softmax.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foreard", &softmax1_forward, "Softmax1 forward pass");
    m.def("backward", &softmax1_backward, "Softmax1 backward pass")
}


