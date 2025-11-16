from setuptools import setup, Extension
import pybind11
import os  # Add this import

# Simple setup without complex compiler flags
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
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3', '-std=c++11']
    ),
]

setup(
    name="raytracer_cpp",
    ext_modules=ext_modules,
)