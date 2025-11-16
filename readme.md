raytracer_project/
├── main.py
├── raytracer_core.py
├── scene.py
├── camera.py
├── denoisers.py
├── requirements.txt
├── setup.py
├── run.py
├── README.md
└── cpp_raytracer/
    ├── setup.py
    ├── raytracer_core.h
    ├── raytracer_core.cpp
    ├── bvh.h
    ├── bvh.cpp
    └── binding.cpp


# Physically Based Ray Tracer with BVH Acceleration

A high-performance ray tracer featuring:
- Physically Based Rendering (PBR) materials
- BVH acceleration for fast ray intersections
- Progressive rendering with real-time preview
- Multiple denoising algorithms
- Dynamic object manipulation
- C++ backend with Python bindings

## Features

- **BVH Acceleration**: 5-50x faster than brute force for complex scenes
- **PBR Materials**: Metallic, roughness, IOR, emission properties
- **Progressive Rendering**: Image quality improves over time
- **Real-time Denoising**: Bilateral, NL-means, Gaussian, Median filters
- **Dynamic Scene**: Move objects in real-time
- **Quality Settings**: Adjust samples, bounces, resolution

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build C++ extension
cd cpp_raytracer
python setup.py build_ext --inplace
cd ..

# Run the application
python main.py