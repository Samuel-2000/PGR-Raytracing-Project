# interaction.py - UPDATED VERSION

import numpy as np
import time
import threading
from queue import Queue
from typing import Dict, Optional
import math

from denoiser import Denoiser
from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3, Camera

class RayTracerInteraction:
    """C++ Ray Tracer with Full Interactive Controls - FIXED VERSION"""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        
        # Initialize C++ ray tracer
        self.ray_tracer = RayTracer()
        self.scene = self.create_interactive_scene()
        self.ray_tracer.set_scene(self.scene)
        
        # Get camera reference
        self.camera = self.ray_tracer.get_camera()
        
        # Initialize camera to safe position
        self.camera.position = Vector3(0, 2, 5)  # Safer starting position
        self.camera.target = Vector3(0, 0, -1)
        self.camera.up = Vector3(0, 1, 0)
        self.camera.fov = 45.0
        
        # Rendering state
        self.is_rendering = False
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Settings with safer defaults
        self.settings = {
            'max_samples': 32,  # Reduced for faster feedback
            'samples_per_batch': 8,  # Reduced for stability
            'max_depth': 4,  # Reduced for stability
            'exposure': 1.5,
            'enhance_image': True,
            'show_denoisers': False,
            'selected_denoisers': ['bilateral'],
            'selected_object': 1,  # Start with first interactive object
            'move_speed': 0.3,  # Slower movement
            'camera_move_speed': 0.05,  # Slower camera movement
        }
        
        # Object manipulation state
        self.dragging = False
        self.camera_dragging = False
        self.last_mouse_pos = None
        
        # Thread safety
        self.render_lock = threading.Lock()
        
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
        ground.name = "Ground"
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
        idx = self.settings['selected_object']
        if 0 <= idx < len(self.scene.spheres):
            return self.scene.spheres[idx]
        return None
    
    def select_object_by_click(self, x: float, y: float) -> bool:
        """Select object by screen coordinates - FIXED VERSION"""
        try:
            # Use C++ ray casting for object selection
            object_id = self.ray_tracer.select_object(x, y, self.width, self.height)
            if object_id >= 0 and object_id < len(self.scene.spheres):
                self.settings['selected_object'] = object_id
                obj = self.get_selected_object()
                if obj:
                    print(f"Selected: {obj.name} (ID: {obj.object_id})")
                    return True
        except Exception as e:
            print(f"Object selection error: {e}")
        return False

    def move_object(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move selected object - FIXED VERSION"""
        with self.render_lock:
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:  # Don't move ground
                speed = self.settings['move_speed']
                obj.center.x += dx * speed
                obj.center.y += dy * speed  
                obj.center.z += dz * speed
                
                # Add bounds checking
                obj.center.x = max(-8, min(8, obj.center.x))
                obj.center.y = max(0.2, min(8, obj.center.y))
                obj.center.z = max(-8, min(2, obj.center.z))
                
                self.scene.build_bvh()
                self.restart_rendering()
                print(f"Moved {obj.name} to ({obj.center.x:.1f}, {obj.center.y:.1f}, {obj.center.z:.1f})")

    def move_camera(self, dx: float, dy: float, dz: float):
        """Move camera in world space - FIXED VERSION"""
        with self.render_lock:
            speed = self.settings['camera_move_speed']
            
            # Calculate movement vectors based on camera orientation
            forward = (self.camera.target - self.camera.position).normalize()
            right = forward.cross(self.camera.up).normalize()
            up = self.camera.up.normalize()
            
            # Apply movement
            move_vector = (right * dx + up * dy + forward * dz) * speed
            self.camera.position = self.camera.position + move_vector
            self.camera.target = self.camera.target + move_vector
            
            self.restart_rendering()

    def rotate_camera(self, dx: float, dy: float):
        """Rotate camera around target - SIMPLIFIED VERSION"""
        with self.render_lock:
            # Simple rotation - adjust camera position around target
            angle_x = dx * 0.5  # Reduced sensitivity
            angle_y = dy * 0.5
            
            # Get vector from target to camera
            vec = self.camera.position - self.camera.target
            
            # Rotate around Y axis (horizontal)
            cos_x = math.cos(angle_x)
            sin_x = math.sin(angle_x)
            new_x = vec.x * cos_x - vec.z * sin_x
            new_z = vec.x * sin_x + vec.z * cos_x
            vec = Vector3(new_x, vec.y, new_z)
            
            # Rotate around X axis (vertical) with limits
            cos_y = math.cos(angle_y)
            sin_y = math.sin(angle_y)
            new_y = vec.y * cos_y - vec.z * sin_y
            new_z = vec.y * sin_y + vec.z * cos_y
            
            # Limit vertical rotation
            if abs(new_y) < 3.0:  # Prevent flipping
                vec = Vector3(vec.x, new_y, new_z)
            
            # Update camera position
            self.camera.position = self.camera.target + vec
            
            self.restart_rendering()

    def update_object_material(self, property_name: str, value: float):
        """Update material property of selected object"""
        obj = self.get_selected_object()
        if obj:
            if property_name == 'albedo':
                # For color, set all channels to the same value for simplicity
                obj.material.albedo = Vector3(value, value, value)
            elif property_name == 'metallic':
                obj.material.metallic = value
            elif property_name == 'roughness':
                obj.material.roughness = value
                
            self.restart_rendering()
            print(f"Updated {obj.name} {property_name} to {value:.2f}")

    def update_light_intensity(self, intensity: float):
        """Update light intensity for selected light"""
        obj = self.get_selected_object()
        if obj and hasattr(obj.material, 'emission'):
            # Check if it's a light (has significant emission)
            emission = obj.material.emission
            is_light = emission.x > 1.0 or emission.y > 1.0 or emission.z > 1.0
            
            if is_light:
                # Preserve color ratios, just scale intensity
                current_max = max(emission.x, emission.y, emission.z)
                if current_max > 0:
                    scale = intensity / current_max
                    obj.material.emission = Vector3(
                        emission.x * scale,
                        emission.y * scale, 
                        emission.z * scale
                    )
                    self.restart_rendering()
                    print(f"Updated {obj.name} intensity to {intensity}")

    def restart_rendering(self):
        """Restart rendering with current settings - FIXED VERSION"""
        with self.render_lock:
            self.is_rendering = False
            time.sleep(0.02)  # Brief pause to ensure thread stops
            
            # Clear accumulated image
            self.accumulated_image = None
            self.total_samples = 0
            self.frame_queue = Queue()
            
            # Force BVH rebuild
            self.scene.build_bvh()
            
            # Start fresh
            self.start_rendering()

    def start_rendering(self):
        """Start progressive rendering"""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.accumulated_image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.total_samples = 0
        
        render_thread = threading.Thread(target=self._render_worker)
        render_thread.daemon = True
        render_thread.start()

    def _render_worker(self):
        """Worker function for rendering - FIXED VERSION"""
        try:
            while (self.is_rendering and 
                   self.total_samples < self.settings['max_samples']):
                
                start_time = time.time()
                
                # Render using C++ ray tracer with thread safety
                with self.render_lock:
                    result = self.ray_tracer.render(
                        self.width, self.height, 
                        self.settings['samples_per_batch'], 
                        self.settings['max_depth']
                    )
                
                if result is None or len(result) == 0:
                    print("Warning: Empty render result")
                    continue
                    
                # Convert to numpy and reshape
                batch_image = np.array(result, dtype=np.float32).reshape(
                    (self.height, self.width, 3)
                )
                render_time = time.time() - start_time
                
                # FIXED: Proper progressive accumulation
                batch_samples = self.settings['samples_per_batch']
                
                if self.total_samples == 0:
                    # First batch
                    self.accumulated_image = batch_image
                    self.total_samples = batch_samples
                else:
                    # Progressive accumulation using running average
                    total_old = self.total_samples
                    total_new = self.total_samples + batch_samples
                    
                    # Weighted average: old * (n/(n+m)) + new * (m/(n+m))
                    weight_old = total_old / total_new
                    weight_new = batch_samples / total_new
                    
                    self.accumulated_image = (
                        self.accumulated_image * weight_old + 
                        batch_image * weight_new
                    )
                    self.total_samples = total_new
                
                # Send frame for display
                if (self.total_samples % self.settings['samples_per_batch'] == 0 or 
                    self.total_samples >= self.settings['max_samples']):
                    self._process_frame_for_display(render_time)
                
                # Small delay to prevent CPU overload
                time.sleep(0.005)
                
        except Exception as e:
            print(f"Rendering error: {e}")
            import traceback
            traceback.print_exc()
        
        # Signal completion
        self.frame_queue.put({'done': True})
        self.is_rendering = False

    def _process_frame_for_display(self, render_time: float):
        """Process frame for display - FIXED VERSION"""
        if self.accumulated_image is None:
            return
            
        # Apply tone mapping to prevent red fog
        display_image = self._tone_map(self.accumulated_image, self.settings['exposure'])
        
        # Enhanced version with contrast adjustment
        enhanced_image = self._enhance_display(display_image) if self.settings['enhance_image'] else display_image
        
        # Denoised versions if enabled
        denoised_images = {}
        if self.settings['show_denoisers'] and self.settings['selected_denoisers']:
            for method in self.settings['selected_denoisers']:
                try:
                    denoised_images[method] = self.denoiser.denoise(display_image, method)
                except Exception as e:
                    print(f"Denoising error with {method}: {e}")
        
        frame_data = {
            'linear': self.accumulated_image.copy(),
            'display': display_image,
            'enhanced': enhanced_image,
            'denoised': denoised_images,
            'samples': self.total_samples,
            'render_time': render_time,
        }
        
        self.frame_queue.put(frame_data)

    def _tone_map(self, image: np.ndarray, exposure: float) -> np.ndarray:
        """Tone mapping to prevent color explosion - FIXED VERSION"""
        # Apply exposure and clamp to prevent infinite growth
        image = image * exposure
        
        # Use Reinhard tone mapping to compress high values
        image = image / (1.0 + image)
        
        # Additional clamping for safety
        return np.clip(image, 0.0, 1.0)

    def _enhance_display(self, image: np.ndarray) -> np.ndarray:
        """Contrast enhancement - FIXED VERSION"""
        # Simple contrast stretch
        min_val = np.percentile(image, 2)
        max_val = np.percentile(image, 98)
        
        if max_val > min_val:
            enhanced = (image - min_val) / (max_val - min_val)
            return np.clip(enhanced, 0, 1)
        else:
            return image

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