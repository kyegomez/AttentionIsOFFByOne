#cupy allows you to compile raw python code into cuda, this a test
import cupy as cp

#softmax

def softmax_one_cupy(x, axis=None):
    #substract the max for stability
    x = x - cp.max(x, axis=axis, keepdims=True)

    #compute exponentials
    exp_x = cp.exp(x)

    #compute the softmax values and add one in the denominator
    return exp_x / (1 + cp.sum(exp_x, axis=axis, keepdims=True))