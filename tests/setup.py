from setuptools import setup, Extension
from torch.utils import cpp_extension

softmax_one_cpp = Extension(
    name="softmax_one_cpp",
    sources=["softmax_one/optimized/softmax_one.cpp", "softmax_one/optimized/binding.cpp"],
    include_dirs=["sotmax_one/include"],
    extra_compile_args=["-std=c++14"]
)

setup(
    name='softmax_one_cpp',
    version='0.1',
    ext_modules=[softmax_one_cpp],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
# # python setup.py install
# import softmax_one_cpp
# def test_softmax_one_cpp():
#     x = torch.randn(10, 5)
#     y = softmax_one_cpp.forward(x, dim=1)
#     assert torch.allclose(y.sum(dim=1), torch.ones(10))
