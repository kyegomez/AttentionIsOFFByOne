#include <pybind11/pybind11.h>
#include <pybind11/st1.h>


namespace py = pybind11;

PYBIND11_MODULE(softmax1_cpp, m) {
    m.def("softmax1", &softmax1, "A softmax implementation in c++")
}

// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` softmax1.cpp -o softmax1_cpp`python3-config --extension-suffix`
// import softmax1_cpp

// x = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
// print(softmax1_cpp.softmax1(x))