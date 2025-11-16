from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import os
import sys

# Windows-specific compiler flags
if os.name == 'nt':  # Windows
    extra_compile_args = ['/O2', '/openmp', '/std:c++17', '/bigobj']
    extra_link_args = []
else:  # Linux/Mac
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp', '-std=c++17']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "raytracer_cpp",
        [
            "binding.cpp",
            "raytracer_core.cpp", 
            "bvh.cpp"
        ],
        include_dirs=["."],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    name="raytracer_cpp",
    ext_modules=ext_modules,
    zip_safe=False,
)