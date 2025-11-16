from setuptools import setup, Extension
import pybind11
import os
import sys

# Determine compiler flags based on platform
if os.name == 'nt':  # Windows
    extra_compile_args = ['/O2', '/std:c++17', '/openmp']
    extra_link_args = []
else:  # Linux/Mac
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp', '-std=c++17']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "raytracer_cpp",
        [
            "binding.cpp",
            "raytracer_core.cpp", 
            "bvh.cpp"
        ],
        include_dirs=[".", pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="raytracer_cpp",
    ext_modules=ext_modules,
    zip_safe=False,
)