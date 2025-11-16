from setuptools import setup, find_packages

setup(
    name="raytracer-pbr",
    version="1.0.0",
    description="Physically Based Ray Tracer with BVH Acceleration",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0", 
        "opencv-python>=4.5.0",
    ],
    python_requires=">=3.8",
)