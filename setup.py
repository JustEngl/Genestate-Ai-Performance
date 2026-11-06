from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='genestate_cuda_ops',
    ext_modules=[
        CUDAExtension('genestate_cuda_ops', [
            'cppcuda/binding.cpp',
            'cppcuda/activation_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)