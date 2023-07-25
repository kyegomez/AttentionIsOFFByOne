#include <pybind11/pybind11.h>
#include <pybind11/st1.h>


namespace py = pybind11;

PYBIND11_MODULE(softmax_one_cpp, m) {
    m.def("softmax_one", &softmax_one, "A softmax implementation in c++")
}

// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` softmax_one.cpp -o softmax_one_cpp`python3-config --extension-suffix`
// import softmax_one_cpp

// x = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
// print(softmax_one_cpp.softmax_one(x))