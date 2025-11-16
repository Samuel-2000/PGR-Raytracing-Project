import numpy as np
import random
from typing import List, Optional
from scene import Scene, Sphere, Material, Vector3

class PythonRayTracer:
    """Python fallback implementation of the ray tracer"""
    
    def __init__(self):
        self.scene = None
    
    def render(self, width: int, height: int, samples_per_pixel: int, max_depth: int) -> np.ndarray:
        if self.scene is None:
            return np.zeros((height, width, 3))
        
        image = np.zeros((height, width, 3))
        
        for j in range(height):
            for i in range(width):
                color = np.zeros(3)
                for _ in range(samples_per_pixel):
                    u = (i + random.random()) / width
                    v = (j + random.random()) / height
                    
                    # Flip v coordinate to fix upside-down image
                    flipped_v = 1.0 - v
                    
                    # Simplified ray tracing for demo
                    color += self._trace_ray(u, flipped_v, max_depth)
                image[j, i] = color / samples_per_pixel
        
        return np.clip(image, 0, 1)
    
    def _trace_ray(self, u: float, v: float, depth: int) -> np.ndarray:
        # Simplified ray tracing implementation
        if depth <= 0:
            return np.array([0.1, 0.1, 0.1])  # Background
        
        # Simple sphere intersection for demo
        for sphere in self.scene.spheres:
            # Simplified intersection check
            if self._intersect_sphere(u - 0.5, v - 0.5, -1, sphere):
                return sphere.material.albedo * 0.8 + self._trace_ray(u, v, depth - 1) * 0.2
        
        return np.array([0.1, 0.1, 0.1])  # Background
    
    def _intersect_sphere(self, x: float, y: float, z: float, sphere: Sphere) -> bool:
        # Simplified sphere intersection
        dx = x - sphere.center.x
        dy = y - sphere.center.y
        dz = z - sphere.center.z
        return dx*dx + dy*dy + dz*dz <= sphere.radius * sphere.radius