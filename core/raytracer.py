# FILE: src/core/raytracer.py
"""
Optimized Ray Tracer Core with GPU Acceleration
"""
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from accelerators.bvh import BVHAccelerator
from accelerators.gpu_accelerator import GPUAccelerator
from .scene import Scene
from .camera import Camera
from .materials import Material, PBRMaterial

logger = logging.getLogger(__name__)

class RayTracer:
    """High-performance ray tracer with multiple acceleration methods"""
    
    def __init__(self, width: int = 800, height: int = 600, use_gpu: bool = True):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        self.scene = None
        self.camera = Camera()
        
        # Acceleration structures
        self.bvh = BVHAccelerator()
        self.gpu_accelerator = GPUAccelerator() if self.use_gpu else None
        
        # Rendering state
        self.accumulator = None
        self.sample_count = 0
        self.is_rendering = False
        
        # Configuration
        self.config = {
            'max_samples': 1024,
            'samples_per_batch': 16,
            'max_bounces': 8,
            'russian_roulette_depth': 3,
            'use_importance_sampling': True,
            'use_next_event_estimation': True,
            'clamp_value': 10.0
        }
        
        logger.info(f"RayTracer initialized: {width}x{height}, GPU: {self.use_gpu}")
    
    def set_scene(self, scene: Scene):
        """Set the scene and build acceleration structures"""
        self.scene = scene
        self.bvh.build(scene.objects)
        
        if self.use_gpu and self.gpu_accelerator:
            self.gpu_accelerator.upload_scene(scene)
        
        self.reset_accumulator()
    
    def set_camera(self, camera: Camera):
        """Set the camera"""
        self.camera = camera
    
    def reset_accumulator(self):
        """Reset the accumulation buffer"""
        self.accumulator = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.sample_count = 0
    
    def render_batch(self, samples: int = 1) -> np.ndarray:
        """Render a batch of samples"""
        if self.scene is None:
            raise ValueError("No scene set for rendering")
        
        start_time = time.time()
        
        if self.use_gpu and self.gpu_accelerator:
            batch_result = self._render_batch_gpu(samples)
        else:
            batch_result = self._render_batch_cpu(samples)
        
        # Update accumulator
        if self.sample_count == 0:
            self.accumulator = batch_result
        else:
            total_samples = self.sample_count + samples
            weight_old = self.sample_count / total_samples
            weight_new = samples / total_samples
            self.accumulator = self.accumulator * weight_old + batch_result * weight_new
        
        self.sample_count += samples
        
        render_time = time.time() - start_time
        logger.debug(f"Rendered {samples} samples in {render_time:.3f}s")
        
        return self.accumulator.copy()
    
    def _render_batch_cpu(self, samples: int) -> np.ndarray:
        """CPU rendering implementation with vectorization"""
        # Generate rays for all pixels and samples
        rays = self.camera.generate_rays(self.width, self.height, samples)
        
        # Vectorized ray tracing
        result = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Process in tiles for better cache performance
        tile_size = 32
        for y in range(0, self.height, tile_size):
            for x in range(0, self.width, tile_size):
                y_end = min(y + tile_size, self.height)
                x_end = min(x + tile_size, self.width)
                
                # Extract tile rays
                tile_rays = rays[y:y_end, x:x_end]
                tile_shape = tile_rays.shape
                
                # Flatten for batch processing
                flat_rays = tile_rays.reshape(-1, 2, 3)  # origin, direction
                
                # Trace rays
                tile_colors = self._trace_rays_batch(flat_rays, samples)
                
                # Reshape back and accumulate
                result[y:y_end, x:x_end] += tile_colors.reshape(
                    tile_shape[0], tile_shape[1], 3
                )
        
        return result / samples
    
    def _render_batch_gpu(self, samples: int) -> np.ndarray:
        """GPU rendering implementation"""
        return self.gpu_accelerator.render_batch(
            self.camera, self.width, self.height, samples, self.config
        )
    
    def _trace_rays_batch(self, rays: np.ndarray, samples: int) -> np.ndarray:
        """Trace a batch of rays (vectorized)"""
        batch_size = rays.shape[0]
        colors = np.zeros((batch_size, 3), dtype=np.float32)
        
        # Current ray states
        current_rays = rays.copy()
        current_throughput = np.ones((batch_size, 3), dtype=np.float32)
        current_depths = np.zeros(batch_size, dtype=np.int32)
        
        active_mask = np.ones(batch_size, dtype=bool)
        
        for bounce in range(self.config['max_bounces']):
            if not np.any(active_mask):
                break
            
            # Intersect rays with scene
            hits = self.bvh.intersect_batch(current_rays[active_mask])
            
            for i, (hit, ray_idx) in enumerate(zip(hits, np.where(active_mask)[0])):
                if hit is None:
                    colors[ray_idx] += current_throughput[ray_idx] * self.scene.background_color
                    active_mask[ray_idx] = False
                    continue
                
                # Shade hit point
                material = hit.material
                hit_point = current_rays[ray_idx, 0] + current_rays[ray_idx, 1] * hit.distance
                normal = hit.normal
                
                # Sample material
                if material.emission.max() > 0:
                    colors[ray_idx] += current_throughput[ray_idx] * material.emission
                
                # Russian roulette termination
                if bounce >= self.config['russian_roulette_depth']:
                    survival_prob = min(current_throughput[ray_idx].max(), 0.95)
                    if np.random.random() > survival_prob:
                        active_mask[ray_idx] = False
                        continue
                    current_throughput[ray_idx] /= survival_prob
                
                # Sample new direction
                new_dir = material.sample_direction(normal, current_rays[ray_idx, 1])
                pdf = material.pdf(normal, current_rays[ray_idx, 1], new_dir)
                
                if pdf > 1e-8:
                    brdf = material.eval(normal, current_rays[ray_idx, 1], new_dir)
                    current_throughput[ray_idx] *= brdf * abs(np.dot(normal, new_dir)) / pdf
                    
                    # Update ray
                    current_rays[ray_idx, 0] = hit_point + normal * 1e-4
                    current_rays[ray_idx, 1] = new_dir
                else:
                    active_mask[ray_idx] = False
            
            current_depths[active_mask] += 1
        
        return colors
    
    def get_progressive_result(self) -> np.ndarray:
        """Get current progressive rendering result with tone mapping"""
        if self.sample_count == 0:
            return np.zeros((self.height, self.width, 3))
        
        # Apply tone mapping
        result = self._tone_map_acces(self.accumulator)
        return np.clip(result, 0, 1)
    
    def _tone_map_acces(self, image: np.ndarray) -> np.ndarray:
        """ACES tone mapping implementation"""
        # ACES approximation
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        
        return np.clip((image * (a * image + b)) / (image * (c * image + d) + e), 0, 1)