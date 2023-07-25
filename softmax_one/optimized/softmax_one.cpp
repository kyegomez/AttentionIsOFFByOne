#include <vector>
#include <cmath>

std::vector<float> softmax_one(std::vector<float> x) {
    // Find max of x
    float max_x = *max_element(x.begin(), x.end());

    // Subtract max_x from each element of x to avoid overflow in exp
    for (auto &elem : x) {
        elem -= max_x;
    }

    // Calculate exp(x) and sum(exp(x))
    std::vector<float> exp_x;
    float sum_exp_x = 0.0;
    for (const auto &elem : x) {
        float exp_elem = std::exp(elem);
        exp_x.push_back(exp_elem);
        sum_exp_x += exp_elem;
    }

    // Apply softmax_one formula
    std::vector<float> softmax1_x;
    for (const auto &elem : exp_x) {
        softmax1_x.push_back(elem / (1 + sum_exp_x));
    }

    return softmax1_x;
}
