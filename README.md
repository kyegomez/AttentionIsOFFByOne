# Quiet Attention - A Novel Modification to Softmax Function for Attention Mechanism

![Logo](link-to-logo)

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


## Getting Started

1. Clone the repository:
```
git clone https://github.com/user/quiet-attention.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Import the model and use it in your Pytorch code:
```python
from quiet_attention import QuietAttention

# Replace traditional softmax attention with Quiet Attention in your transformer model
attention_layer = QuietAttention()
```

## Contributing

We encourage you to contribute to Quiet Attention! Please check out the [Contributing to Quiet Attention guide](CONTRIBUTING.md) for guidelines about how to proceed.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE.md) for additional details.


## Acknowledgements

Special thanks to the researchers and the open-source community who made it possible to build upon their initial work.
