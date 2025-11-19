# interaction.py

import numpy as np

import time
import threading
from queue import Queue
from typing import Dict, Optional

from denoiser import Denoiser
from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3

class RayTracerInteraction:
    """C++ Only Ray Tracer with Interactive Controls"""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        
        # Initialize C++ ray tracer
        self.ray_tracer = RayTracer()
        self.scene = self.create_interactive_scene()
        self.ray_tracer.set_scene(self.scene)
        
        # Rendering state
        self.is_rendering = False
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Settings with defaults
        self.settings = {
            'max_samples': 128,
            'samples_per_batch': 4,
            'max_depth': 8,
            'exposure': 1.5,
            'enhance_image': True,
            'show_denoisers': False,
            'selected_denoisers': ['bilateral'],
            'update_step': 4,
            'selected_object': 0,
            'move_speed': 0.5,
        }
        
        # Object manipulation state
        self.dragging = False
        self.drag_start = None
        
        # Denoiser
        self.denoiser = Denoiser()
        
        print(f"✓ Initialized C++ Ray Tracer: {width}x{height}")

    def create_interactive_scene(self) -> Scene:
        """Create a scene with interactive objects"""
        scene = Scene()
        scene.background_color = Vector3(0.05, 0.05, 0.1)
        
        # Ground
        ground_material = Material()
        ground_material.albedo = Vector3(0.9, 0.9, 0.9)
        ground = Sphere()
        ground.center = Vector3(0, -101, 0)
        ground.radius = 100.0
        ground.material = ground_material
        ground.object_id = 0
        scene.add_sphere(ground)
        
        # Interactive objects
        objects_data = [
            # Spheres
            {"type": "sphere", "pos": (-2.0, 0.5, -6.0), "color": (0.9, 0.1, 0.1), 
             "metal": 0.9, "rough": 0.1, "emission": (0,0,0), "radius": 0.8, "name": "Red Metallic"},
            {"type": "sphere", "pos": (0.0, 0.5, -6.0), "color": (0.1, 0.9, 0.1), 
             "metal": 0.0, "rough": 0.3, "emission": (0,0,0), "radius": 0.8, "name": "Green Dielectric"},
            {"type": "sphere", "pos": (2.0, 0.5, -6.0), "color": (0.1, 0.1, 0.9), 
             "metal": 0.0, "rough": 0.0, "emission": (0,0,0), "radius": 0.8, "name": "Blue Glass"},
            {"type": "sphere", "pos": (-1.0, 2.0, -4.0), "color": (0.9, 0.9, 0.1), 
             "metal": 0.5, "rough": 0.2, "emission": (0,0,0), "radius": 0.6, "name": "Yellow Mixed"},
            
            # Lights
            {"type": "light", "pos": (0, 5, -2), "color": (1, 1, 1), 
             "emission": (15, 15, 12), "radius": 0.8, "name": "Main Light"},
            {"type": "light", "pos": (-3, 3, 0), "color": (1, 1, 1), 
             "emission": (8, 5, 3), "radius": 0.5, "name": "Warm Light"},
            {"type": "light", "pos": (3, 3, 0), "color": (1, 1, 1), 
             "emission": (3, 5, 8), "radius": 0.5, "name": "Cool Light"},
        ]
        
        for i, data in enumerate(objects_data, 1):
            material = Material()
            material.albedo = Vector3(*data["color"])
            
            if data["type"] == "light":
                material.emission = Vector3(*data["emission"])
                material.metallic = 0.0
                material.roughness = 0.1
            else:
                material.metallic = data["metal"]
                material.roughness = data["rough"]
                material.emission = Vector3(*data["emission"])
                material.ior = 1.5
            
            sphere = Sphere()
            sphere.center = Vector3(*data["pos"])
            sphere.radius = data["radius"]
            sphere.material = material
            sphere.object_id = i
            sphere.name = data["name"]
            scene.add_sphere(sphere)
        
        scene.build_bvh()
        return scene

    def get_object_count(self) -> int:
        """Get number of interactive objects (excluding ground)"""
        return len(self.scene.spheres) - 1

    def get_selected_object(self) -> Optional[Sphere]:
        """Get currently selected object"""
        idx = self.settings['selected_object'] + 1  # +1 to skip ground
        if 0 < idx < len(self.scene.spheres):
            return self.scene.spheres[idx]
        return None

    def move_object(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move selected object"""
        obj = self.get_selected_object()
        if obj:
            speed = self.settings['move_speed']
            obj.center.x += dx * speed
            obj.center.y += dy * speed
            obj.center.z += dz * speed
            self.scene.build_bvh()
            self.restart_rendering()

    def update_object_material(self, property_name: str, value: float):
        """Update material property of selected object"""
        obj = self.get_selected_object()
        if obj:
            if property_name == 'albedo':
                current = obj.material.albedo
                obj.material.albedo = Vector3(value, current.y, current.z)
            elif property_name == 'emission':
                current = obj.material.emission
                obj.material.emission = Vector3(value, current.y, current.z)
            elif hasattr(obj.material, property_name):
                setattr(obj.material, property_name, value)
            self.restart_rendering()

    def update_light_intensity(self, intensity: float):
        """Update light intensity for selected light"""
        obj = self.get_selected_object()
        if obj and obj.material.emission.x > 0:  # Check if it's a light
            scale = intensity / max(obj.material.emission.x, 1.0)
            current = obj.material.emission
            obj.material.emission = Vector3(
                current.x * scale,
                current.y * scale, 
                current.z * scale
            )
            self.restart_rendering()

    def restart_rendering(self):
        """Restart rendering with current settings"""
        if self.is_rendering:
            self.is_rendering = False
            time.sleep(0.1)
        
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        self.start_rendering()

    def start_rendering(self):
        """Start progressive rendering"""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.accumulated_image = np.zeros((self.height, self.width, 3))
        self.total_samples = 0
        
        render_thread = threading.Thread(target=self._render_worker)
        render_thread.daemon = True
        render_thread.start()

    def _render_worker(self):
        """Worker function for rendering"""
        while self.is_rendering and self.total_samples < self.settings['max_samples']:
            try:
                start_time = time.time()
                
                # Render using C++ ray tracer
                result = self.ray_tracer.render(
                    self.width, self.height, 
                    self.settings['samples_per_batch'], 
                    self.settings['max_depth']
                )
                batch_image = np.array(result).reshape((self.height, self.width, 3))
                
                render_time = time.time() - start_time
                
                # Update accumulated image
                self.total_samples += self.settings['samples_per_batch']
                
                if self.total_samples == self.settings['samples_per_batch']:
                    self.accumulated_image = batch_image
                else:
                    weight_old = (self.total_samples - self.settings['samples_per_batch']) / self.total_samples
                    weight_new = self.settings['samples_per_batch'] / self.total_samples
                    self.accumulated_image = self.accumulated_image * weight_old + batch_image * weight_new
                
                # Send frame if we've reached the update step
                if (self.total_samples % self.settings['update_step'] == 0 or 
                    self.total_samples >= self.settings['max_samples']):
                    self._process_frame_for_display(render_time)
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Rendering error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        self.frame_queue.put({'done': True})
        self.is_rendering = False

    def _process_frame_for_display(self, render_time: float):
        """Process frame for display with tone mapping and denoising"""
        linear_image = np.clip(self.accumulated_image, 0, 10)
        
        # Tone mapping
        display_image = self._tone_map(linear_image, self.settings['exposure'])
        
        # Enhanced version
        enhanced_image = self._enhance_display(linear_image) if self.settings['enhance_image'] else display_image
        
        # Denoised versions if enabled
        denoised_images = {}
        if self.settings['show_denoisers'] and self.settings['selected_denoisers']:
            for method in self.settings['selected_denoisers']:
                try:
                    denoised_images[method] = self.denoiser.denoise(display_image, method)
                except Exception as e:
                    print(f"Denoising error with {method}: {e}")
        
        frame_data = {
            'linear': linear_image,
            'display': display_image,
            'enhanced': enhanced_image,
            'denoised': denoised_images,
            'samples': self.total_samples,
            'render_time': render_time,
        }
        
        self.frame_queue.put(frame_data)

    def _tone_map(self, image: np.ndarray, exposure: float) -> np.ndarray:
        """Simple Reinhard tone mapping"""
        image = image * exposure
        return image / (1.0 + image)

    def _enhance_display(self, image: np.ndarray) -> np.ndarray:
        """Contrast enhancement"""
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:
            enhanced = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            enhanced = image
        return enhanced

    def stop_rendering(self):
        """Stop rendering"""
        self.is_rendering = False

    def has_frames(self) -> bool:
        """Check if frames are available"""
        return not self.frame_queue.empty()

    def get_frame(self) -> Optional[Dict]:
        """Get next frame"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None