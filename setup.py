from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='ugp',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='ugp.ext',
            sources=[
                'ugp/extensions/extra/cloud/cloud.cpp',
                'ugp/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'ugp/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'ugp/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'ugp/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'ugp/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
