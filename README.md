<div class='center'>

# Join Agora
Agora is devoted to advancing Humanity with extremely advanced AI Research

<img src="partnership-banner.png" alt="Agora banner" width="700" height="500" >

Join our discord here!
![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)

</div>




# Quiet Attention - A Novel Modification to Softmax Function for Attention Mechanism

```math
(softmax_one(x))_i = exp(x_i) / (1 + sum(exp(x_j) for all j))
```

Attention mechanism has been a groundbreaking innovation in deep learning, and forms the backbone of the Transformer models, which powers the state-of-the-art language models like GPT4 and LLAMA. However, there is a persistent off-by-one bug in the traditional attention mechanism that can make the models harder to compress and deploy.

Introducing Quiet Attention, an innovative tweak to the traditional softmax function, allowing the attention heads to express 'no preference' and remain quiet. The slight adjustment to the denominator allows the vector to tend to zero if it prefers, rather than forcing the attention head to make an annotation.

[This is a paper by Evan Miller, here's the link](https://www.evanmiller.org/attention-is-off-by-one.html)


## Formula

Here's the modified formula for the softmax function, also referred to as "Softmax1" or "Quiet Attention" formula:

```math
(softmax_one(x))_i = exp(x_i) / (1 + sum(exp(x_j) for all j))
```

## Architecture

The critical difference between Softmax1 and traditional softmax lies in their negative limit behavior. In a scenario where all the entries in a vector are significantly less than zero and the model wants to avoid an annotation altogether, softmax_one allows it, unlike softmax.

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

A benchmarking suite is included to compare the performance of the `softmax_one` function with the PyTorch native `softmax` function. We provide metrics across different tensor sizes to understand how they perform under varying loads.

To run the benchmarks, use the following command:

```bash
python benchmark.py
```

You can find the results in the `benchmarks/results/` directory. The results include execution time and memory usage for each function across a variety of tensor sizes.

## Usage

You can use the Softmax1 function just like you would use the traditional softmax function. Here's a simple example:

```python
import torch
from softmax_one.softmax_one import softmax_one

x = torch.randn(5)
y = softmax_one(x, dim=0)
```


## Contributions

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or find any bugs.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.


# Experiments 

It's really slow in basic python I will implement it in cuda

```
INFO:root:Running benchmark for tensor size (10, 10)...
INFO:root:F.softmax time: 0.0022182464599609375 s
INFO:root:softmax_one time: 0.04441571235656738 s
INFO:root:Running benchmark for tensor size (100, 100)...
INFO:root:F.softmax time: 0.01704573631286621 s
INFO:root:softmax_one time: 0.07482171058654785 s
INFO:root:Running benchmark for tensor size (1000, 1000)...
INFO:root:F.softmax time: 0.060335397720336914 s
INFO:root:softmax_one time: 3.0616047382354736 s
INFO:root:Running benchmark for tensor size (10000, 10000)...
INFO:root:F.softmax time: 52.80402970314026 s
INFO:root:softmax_one time: 128.78072810173035 s
INFO:root:Chart display is off.

```