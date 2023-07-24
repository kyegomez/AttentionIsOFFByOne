<div class='center'>

# Join Agora
Agora is devoted to advancing Humanity with extremely advanced AI Research

![Logo](partnership-banner.png)

Join our discord here!
![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)


</div>




# Quiet Attention - A Novel Modification to Softmax Function for Attention Mechanism

Attention mechanism has been a groundbreaking innovation in deep learning, and forms the backbone of the Transformer models, which powers the state-of-the-art language models like GPT and BERT. However, there is a persistent off-by-one bug in the traditional attention mechanism that can make the models harder to compress and deploy.

Introducing Quiet Attention, an innovative tweak to the traditional softmax function, allowing the attention heads to express 'no preference' and remain quiet. The slight adjustment to the denominator allows the vector to tend to zero if it prefers, rather than forcing the attention head to make an annotation.

## Formula

Here's the modified formula for the softmax function, also referred to as "Softmax1" or "Quiet Attention" formula:

```math
(softmax1(x))_i = exp(x_i) / (1 + sum(exp(x_j) for all j))
```

## Working

The critical difference between Softmax1 and traditional softmax lies in their negative limit behavior. In a scenario where all the entries in a vector are significantly less than zero and the model wants to avoid an annotation altogether, softmax1 allows it, unlike softmax.

Softmax1 essentially provides an 'escape hatch' when the attention head wants to remain quiet. The total output weight from Softmax1 varies based on the vector input, as opposed to softmax, which always emits the same total weight. This can significantly improve the model's performance, especially when dealing with noisy inputs.


## Installation

Clone the repository:

```
git clone https://github.com/kyegomez/AttentionIsOFFByOne.git
cd AttentionIsOFFByOne
```

## Unit Tests

This repository contains extensive unit tests that aim to cover all possible scenarios and ensure the reliability of the solution. You can run the tests using the following command:

```bash
python -m unittest test.py
```

## Benchmarks

A benchmarking suite is included to compare the performance of the `softmax1` function with the PyTorch native `softmax` function. We provide metrics across different tensor sizes to understand how they perform under varying loads.

To run the benchmarks, use the following command:

```bash
python benchmark.py
```

You can find the results in the `benchmarks/results/` directory. The results include execution time and memory usage for each function across a variety of tensor sizes.

## Usage

You can use the Softmax1 function just like you would use the traditional softmax function. Here's a simple example:

```python
import torch
from attention import softmax1

x = torch.randn(5)
y = softmax1(x, dim=0)
```

For more detailed examples and use cases, refer to the `examples/` directory.

## Contributions

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or find any bugs.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
