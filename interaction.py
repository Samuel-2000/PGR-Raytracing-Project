# interaction.py

import numpy as np
import time
import threading
from queue import Queue
from typing import Dict, Optional

from denoiser import Denoiser
from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3, Camera

class RayTracerInteraction:
    """C++ Ray Tracer with Full Interactive Controls"""
    
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        
        # Initialize C++ ray tracer
        self.ray_tracer = RayTracer()
        self.scene = self.create_interactive_scene()
        self.ray_tracer.set_scene(self.scene)
        
        # Get camera reference
        self.camera = self.ray_tracer.get_camera()
        
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
            'selected_object': 1,  # Start with first interactive object
            'move_speed': 0.5,
            'camera_move_speed': 0.1,
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



    def select_object_by_click(self, x: float, y: float) -> bool:
        """Select object by screen coordinates (0-1 normalized)"""
        # Convert normalized coordinates to ray direction
        ray_dir = self._screen_to_world_ray(x, y)
        ray = Ray(self.camera_position, ray_dir)
        
        # Find closest hit object
        closest_t = float('inf')
        selected_obj = None
        selected_index = -1
        
        for i, sphere in enumerate(self.scene.spheres):
            rec = HitRecord()
            if sphere.hit(ray, 0.001, 1000.0, rec):
                if rec.t < closest_t:
                    closest_t = rec.t
                    selected_obj = sphere
                    selected_index = i
        
        if selected_index >= 0:
            self.settings['selected_object'] = selected_index
            print(f"Selected object: {selected_obj.name if hasattr(selected_obj, 'name') else f'Object {selected_index}'}")
            return True
        return False
    

    def _screen_to_world_ray(self, x: float, y: float) -> Vector3:
        """Convert screen coordinates to world space ray direction"""
        # Convert from [0,1] to [-1,1] and flip Y
        ndc_x = x * 2.0 - 1.0
        ndc_y = (1.0 - y) * 2.0 - 1.0  # Flip Y
        
        aspect_ratio = self.width / self.height
        
        # Calculate ray direction in camera space
        tan_fov = math.tan(math.radians(self.camera_fov / 2.0))
        ray_dir_camera = Vector3(
            ndc_x * aspect_ratio * tan_fov,
            ndc_y * tan_fov,
            -1.0  # Looking along negative Z
        )
        
        # Transform to world space (simplified - assuming camera looks at -Z)
        return ray_dir_camera.normalize()


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
        """Select object by screen coordinates using C++ ray casting"""
        try:
            object_id = self.ray_tracer.select_object(x, y, self.width, self.height)
            if object_id >= 0:
                self.settings['selected_object'] = object_id
                obj = self.get_selected_object()
                if obj:
                    print(f"Selected object: {obj.name}")
                    return True
        except Exception as e:
            print(f"Object selection error: {e}")
        return False

    def move_object(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move selected object"""
        with self.render_lock:
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:  # Don't move ground
                speed = self.settings['move_speed'] * 0.5
                obj.center.x += dx * speed
                obj.center.y += dy * speed  
                obj.center.z += dz * speed
                
                # Add bounds checking
                obj.center.x = max(-10, min(10, obj.center.x))
                obj.center.y = max(0.1, min(10, obj.center.y))
                obj.center.z = max(-15, min(5, obj.center.z))
                
                self.scene.build_bvh()
                self.restart_rendering()


    def move_camera(self, dx: float, dy: float, dz: float):
        """Move camera in world space using C++ camera"""
        with self.render_lock:
            speed = self.settings['camera_move_speed'] * 0.5
            delta = Vector3(dx * speed, dy * speed, dz * speed)
            self.ray_tracer.move_camera(delta)
            print(f"Camera moved to: ({self.camera.position.x:.2f}, {self.camera.position.y:.2f}, {self.camera.position.z:.2f})")
            self.restart_rendering()

    def rotate_camera(self, dx: float, dy: float):
        """Rotate camera around target"""
        with self.render_lock:
            # Simple rotation - adjust camera position around target
            forward = Vector3(
                self.camera.target.x - self.camera.position.x,
                self.camera.target.y - self.camera.position.y, 
                self.camera.target.z - self.camera.position.z
            )
            
            # Calculate right vector
            right = forward.cross(self.camera.up).normalize()
            
            # Rotate around up axis (yaw)
            yaw_angle = dx * 0.01
            rotation_yaw = Vector3(
                forward.x * math.cos(yaw_angle) - forward.z * math.sin(yaw_angle),
                forward.y,
                forward.x * math.sin(yaw_angle) + forward.z * math.cos(yaw_angle)
            )
            
            # Rotate around right axis (pitch)
            pitch_angle = dy * 0.01
            rotation_pitch = Vector3(
                rotation_yaw.x,
                rotation_yaw.y * math.cos(pitch_angle) - rotation_yaw.z * math.sin(pitch_angle),
                rotation_yaw.y * math.sin(pitch_angle) + rotation_yaw.z * math.cos(pitch_angle)
            )
            
            # Update camera position
            distance = forward.length()
            new_forward = rotation_pitch.normalize()
            self.camera.position = Vector3(
                self.camera.target.x - new_forward.x * distance,
                self.camera.target.y - new_forward.y * distance,
                self.camera.target.z - new_forward.z * distance
            )
            
            self.restart_rendering()

    def update_object_material(self, property_name: str, value: float):
        """Update material property of selected object"""
        obj = self.get_selected_object()
        if obj:
            if property_name == 'albedo':
                obj.material.albedo = Vector3(value, value, value)
            elif property_name == 'emission':
                current = obj.material.emission
                obj.material.emission = Vector3(value, current.y, current.z)
            elif hasattr(obj.material, property_name):
                setattr(obj.material, property_name, value)
            self.restart_rendering()

    def update_light_intensity(self, intensity: float):
        """Update light intensity for selected light"""
        obj = self.get_selected_object()
        if obj and hasattr(obj.material, 'emission'):
            emission = obj.material.emission
            is_light = emission.x > 0 or emission.y > 0 or emission.z > 0
            
            if is_light:
                current_total = emission.x + emission.y + emission.z
                if current_total > 0:
                    ratio_x = emission.x / current_total
                    ratio_y = emission.y / current_total  
                    ratio_z = emission.z / current_total
                    
                    total_intensity = intensity
                    obj.material.emission = Vector3(
                        ratio_x * total_intensity,
                        ratio_y * total_intensity,
                        ratio_z * total_intensity
                    )
                    print(f"Updated light intensity to {intensity}")
                    self.restart_rendering()

    def restart_rendering(self):
        """Restart rendering with current settings - IMPROVED VERSION"""
        with self.render_lock:
            if self.is_rendering:
                self.is_rendering = False
                # Give thread more time to stop cleanly
                time.sleep(0.05)
            
            # Clear accumulated image
            self.accumulated_image = None
            self.total_samples = 0
            self.frame_queue = Queue()
            
            # Force BVH rebuild to ensure consistency
            self.scene.build_bvh()
            
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
        """Worker function for rendering - IMPROVED VERSION"""
        # Ensure BVH is built before rendering
        with self.render_lock:
            self.scene.build_bvh()
        
        while self.is_rendering and self.total_samples < self.settings['max_samples']:
            try:
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
                    
                batch_image = np.array(result).reshape((self.height, self.width, 3))
                render_time = time.time() - start_time
                
                # Update accumulated image with proper weighting
                self.total_samples += self.settings['samples_per_batch']
                
                if self.accumulated_image is None:
                    self.accumulated_image = batch_image.copy()
                else:
                    weight_old = (self.total_samples - self.settings['samples_per_batch']) / self.total_samples
                    weight_new = self.settings['samples_per_batch'] / self.total_samples
                    self.accumulated_image = self.accumulated_image * weight_old + batch_image * weight_new
                
                # Send frame for display
                if (self.total_samples % self.settings['update_step'] == 0 or 
                    self.total_samples >= self.settings['max_samples']):
                    self._process_frame_for_display(render_time)
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
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