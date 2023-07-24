import unittest
import torch
from softmax_one.softmax_one import ScaledDotProductAttention

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.module = ScaledDotProductAttention(dropout=0.1)
        self.q = torch.rand(16, 10, 64) #16 batches 10 queries of size 64
        self.k = torch.rand(16, 10, 64) #16 batches of 10 keys of size 64
        self.v = torch.rand(16, 10, 64) #16 batches of 10 values each of size 64
    
    def test_output_shape(self):
        output, _ = self.module(self.q, self.k, self.v)
        self.assertEqual(output.shape, (16, 10, 64))

if __name__ == '__main__':
    unittest.main()