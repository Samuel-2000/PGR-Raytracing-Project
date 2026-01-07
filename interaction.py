import numpy as np
import time
import threading
from queue import Queue
from typing import Dict, Optional, Tuple
import math
from enum import Enum
import cv2

from denoiser import Denoiser
from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3, Camera
from utils import FrameRateLimiter

class RenderMode(Enum):
    """Rendering modes for different interaction scenarios"""
    RAYTRACING = 0
    SILHOUETTE = 1
    WIREFRAME = 2

class Matrix3:
    """Simple 3x3 matrix for camera rotations"""
    
    @staticmethod
    def rotation_y(angle: float) -> 'Matrix3':
        c, s = math.cos(angle), math.sin(angle)
        return Matrix3([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_axis(axis: Vector3, angle: float) -> 'Matrix3':
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = axis.x, axis.y, axis.z
        
        return Matrix3([
            [c + (1-c)*x*x, (1-c)*x*y - s*z, (1-c)*x*z + s*y],
            [(1-c)*x*y + s*z, c + (1-c)*y*y, (1-c)*y*z - s*x],
            [(1-c)*x*z - s*y, (1-c)*y*z + s*x, c + (1-c)*z*z]
        ])
    
    def __init__(self, data):
        self.data = data
    
    def __mul__(self, vec: Vector3) -> Vector3:
        m = self.data
        return Vector3(
            m[0][0]*vec.x + m[0][1]*vec.y + m[0][2]*vec.z,
            m[1][0]*vec.x + m[1][1]*vec.y + m[1][2]*vec.z,
            m[2][0]*vec.x + m[2][1]*vec.y + m[2][2]*vec.z
        )

class RayTracerInteraction:
    """C++ Ray Tracer with Complete Interactive Controls"""
    
    def __init__(self, width: int = 640, height: int = 480, debug_mode: bool = False):
        self.width = width
        self.height = height
        
        # Initialize C++ ray tracer
        self.ray_tracer = RayTracer()
        self.scene = self.create_interactive_scene()
        self.ray_tracer.set_scene(self.scene)
        
        # Get camera reference
        self.camera = self.ray_tracer.get_camera()
        
        # Initialize camera
        self.camera.position = Vector3(0, 2, 5)
        self.camera.target = Vector3(0, 0, -1)
        self.camera.up = Vector3(0, 1, 0)
        self.camera.fov = 45.0
        
        # Camera control state
        self.camera_keys_pressed = {
            'forward': False,  # W
            'backward': False,  # S
            'left': False,  # A
            'right': False,  # D
            'up': False,  # Space/Shift
            'down': False,  # Ctrl
        }
        self.camera_rotating = False
        self.last_mouse_pos = None
        self.camera_last_update_time = 0
        self.camera_update_delay = 0.05  # 50ms delay between updates
        
        # Store previous mode before camera interaction
        self.previous_render_mode = RenderMode.RAYTRACING
        
        # Camera orientation frame
        self.update_camera_frame()
        
        # Object dragging state
        self.dragging_object = False
        self.selected_object_id = -1
        self.drag_start_pos = None
        self.drag_start_object_pos = None
        self.lock_x = self.lock_y = self.lock_z = False
        
        # Rendering state
        self.render_mode = RenderMode.RAYTRACING
        self.is_rendering = False
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Buffers for fast rendering modes
        self.silhouette_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.wireframe_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Settings
        self.settings = {
            'max_samples': 32,
            'samples_per_batch': 8,
            'max_depth': 4,
            'exposure': 1.5,
            'enhance_image': True,
            'show_denoisers': False,
            'selected_denoisers': ['bilateral'],
            'selected_object': 1,
            'move_speed': 0.3,
            'camera_move_speed': 0.1,
            'camera_rotate_speed': 0.5,
        }
        
        # Thread safety
        self.render_lock = threading.RLock()
        
        # Denoiser
        self.denoiser = Denoiser()
        
        # Camera movement thread
        self.camera_move_active = True
        self.camera_move_thread = threading.Thread(target=self._camera_move_worker, daemon=True)
        self.camera_move_thread.start()
        
        print(f"✓ Initialized Interactive Ray Tracer ({width}x{height})")
        print(f"  Controls: WASD+Space/Ctrl to move, Right Mouse to rotate")
        print(f"  Object Drag: Hold X/Y/Z + Left Click + Drag")
    
    def update_camera_frame(self):
        """Update camera orientation vectors"""
        # Forward vector (camera to target)
        self.camera_forward = (self.camera.target - self.camera.position).normalize()
        
        # Right vector (perpendicular to forward and world up)
        world_up = Vector3(0, 1, 0)
        self.camera_right = self.camera_forward.cross(world_up).normalize()
        if self.camera_right.length() == 0:
            self.camera_right = Vector3(1, 0, 0)
        
        # Up vector (perpendicular to forward and right)
        self.camera_up = self.camera_right.cross(self.camera_forward).normalize()
    
    def _camera_move_worker(self):
        """Continuous camera movement worker thread with frame limiting"""
        # Add import at top: from utils import FrameRateLimiter
        limiter = FrameRateLimiter(30)  # Limit to 30 FPS
        
        while self.camera_move_active:
            try:
                # Only process if we should update based on frame rate
                if any(self.camera_keys_pressed.values()) and limiter.should_update():
                    self._process_camera_movement()
                    limiter.update()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Camera worker error: {e}")
                time.sleep(0.1)
    
    def _process_camera_movement(self):
        """Process continuous camera movement with bounds checking"""
        with self.render_lock:
            move_vector = Vector3(0, 0, 0)
            speed = self.settings['camera_move_speed']
            
            # Forward/backward movement
            if self.camera_keys_pressed['forward']:
                move_vector = move_vector + self.camera_forward * speed
            if self.camera_keys_pressed['backward']:
                move_vector = move_vector - self.camera_forward * speed
            
            # Left/right strafing
            if self.camera_keys_pressed['left']:
                move_vector = move_vector - self.camera_right * speed
            if self.camera_keys_pressed['right']:
                move_vector = move_vector + self.camera_right * speed
            
            # Up/down movement
            if self.camera_keys_pressed['up']:
                move_vector = move_vector + Vector3(0, speed, 0)
            if self.camera_keys_pressed['down']:
                move_vector = move_vector - Vector3(0, speed, 0)
            
            if move_vector.length() > 0:
                # Apply movement
                self.camera.position = self.camera.position + move_vector
                self.camera.target = self.camera.target + move_vector
                
                # Apply bounds to camera
                self.camera.position.x = max(-20, min(20, self.camera.position.x))
                self.camera.position.y = max(0.1, min(20, self.camera.position.y))
                self.camera.position.z = max(-20, min(20, self.camera.position.z))
                
                self.update_camera_frame()
                
                # Store previous mode and switch to wireframe
                if self.render_mode == RenderMode.RAYTRACING:
                    self.previous_render_mode = RenderMode.RAYTRACING
                    self.render_mode = RenderMode.WIREFRAME
                
                # Force a display update
                self._process_frame_for_display(0.05)
    
    def start_camera_rotation(self, x: float, y: float):
        """Start camera rotation with mouse"""
        with self.render_lock:
            self.camera_rotating = True
            self.last_mouse_pos = (x, y)
            
            # Store previous mode
            if self.render_mode == RenderMode.RAYTRACING:
                self.previous_render_mode = RenderMode.RAYTRACING
            
            # Always switch to wireframe during rotation for responsiveness
            self.render_mode = RenderMode.WIREFRAME
            print(f"Camera rotation started, previous mode: {self.previous_render_mode}")
    
    def update_camera_rotation(self, dx: float, dy: float):
        """Update camera rotation based on mouse movement"""
        if not self.camera_rotating:
            return
        
        with self.render_lock:
            sensitivity = self.settings['camera_rotate_speed']
            yaw = -dx * sensitivity
            pitch = -dy * sensitivity
            
            # Limit pitch to prevent flipping
            pitch = max(-1.5, min(1.5, pitch))
            
            # Current orientation
            forward = (self.camera.target - self.camera.position).normalize()
            right = forward.cross(Vector3(0, 1, 0)).normalize()
            
            # Yaw rotation (around world up)
            yaw_rot = Matrix3.rotation_y(yaw)
            forward = yaw_rot * forward
            
            # Pitch rotation (around camera right)
            if abs(pitch) > 0.001:
                pitch_rot = Matrix3.rotation_axis(right, pitch)
                forward = pitch_rot * forward
            
            # Update camera
            self.camera.target = self.camera.position + forward
            self.update_camera_frame()
            
            # Force display update
            self._process_frame_for_display(0.05)
    
    def stop_camera_rotation(self):
        """Stop camera rotation and return to previous mode"""
        with self.render_lock:
            self.camera_rotating = False
            self.last_mouse_pos = None
            
            # Return to previous mode if it was raytracing
            if self.previous_render_mode == RenderMode.RAYTRACING and self.render_mode == RenderMode.WIREFRAME:
                self.render_mode = RenderMode.RAYTRACING
                print("Camera rotation stopped, returning to ray tracing")
                # Trigger a restart of rendering
                self.restart_rendering()
            else:
                # Stay in current mode (wireframe or silhouette)
                print(f"Camera rotation stopped, staying in {self.render_mode}")
    
    def set_camera_key_state(self, key: str, state: bool):
        """Update camera key state with debouncing"""
        if key in self.camera_keys_pressed:
            old_state = self.camera_keys_pressed[key]
            self.camera_keys_pressed[key] = state
            
            # If key was just pressed, store previous mode
            if state and not old_state and self.render_mode == RenderMode.RAYTRACING:
                self.previous_render_mode = RenderMode.RAYTRACING
            
            # If any movement key is pressed, switch to wireframe for responsiveness
            if state and self.render_mode == RenderMode.RAYTRACING:
                self.render_mode = RenderMode.WIREFRAME
                self._process_frame_for_display(0.05)
            
            # If all keys released and not rotating, return to ray tracing
            elif not state and not any(self.camera_keys_pressed.values()) and not self.camera_rotating:
                if self.previous_render_mode == RenderMode.RAYTRACING:
                    self.render_mode = RenderMode.RAYTRACING
                    self.restart_rendering()
                # Reset the last update time
                self.camera_last_update_time = 0
    
    def create_interactive_scene(self) -> Scene:
        """Create a scene with interactive objects"""
        scene = Scene()
        scene.background_color = Vector3(0.05, 0.05, 0.1)
        
        # Ground
        ground_material = Material()
        ground_material.albedo = Vector3(0.9, 0.9, 0.9)
        ground = Sphere()
        ground.center = Vector3(0, -100.5, 0)
        ground.radius = 100.0
        ground.material = ground_material
        ground.object_id = 0
        ground.name = "Ground"
        scene.add_sphere(ground)
        
        # Interactive objects
        objects_data = [
            # Main spheres
            {"pos": (-2.0, 0.5, -3.0), "color": (0.9, 0.1, 0.1), 
             "metal": 0.9, "rough": 0.1, "radius": 0.5, "name": "Red Metallic"},
            {"pos": (0.0, 0.5, -3.0), "color": (0.1, 0.9, 0.1), 
             "metal": 0.0, "rough": 0.3, "radius": 0.5, "name": "Green Dielectric"},
            {"pos": (2.0, 0.5, -3.0), "color": (0.1, 0.1, 0.9), 
             "metal": 0.0, "rough": 0.0, "radius": 0.5, "name": "Blue Glass"},
            
            # Additional spheres
            {"pos": (-1.0, 0.3, -1.5), "color": (0.9, 0.9, 0.1), 
             "metal": 0.5, "rough": 0.2, "radius": 0.3, "name": "Yellow Mixed"},
            {"pos": (1.0, 0.3, -1.5), "color": (0.9, 0.1, 0.9), 
             "metal": 0.2, "rough": 0.8, "radius": 0.3, "name": "Purple Rough"},
            
            # Lights
            {"pos": (0, 3, -1), "color": (1, 1, 1), "emission": (10, 10, 8),
             "metal": 0.0, "rough": 0.1, "radius": 0.3, "name": "Main Light"},
            {"pos": (-2, 2, 0), "color": (1, 1, 1), "emission": (5, 3, 2),
             "metal": 0.0, "rough": 0.1, "radius": 0.2, "name": "Warm Light"},
            {"pos": (2, 2, 0), "color": (1, 1, 1), "emission": (2, 3, 5),
             "metal": 0.0, "rough": 0.1, "radius": 0.2, "name": "Cool Light"},
        ]
        
        for i, data in enumerate(objects_data, 1):
            material = Material()
            material.albedo = Vector3(*data["color"])
            material.metallic = data["metal"]
            material.roughness = data["rough"]
            
            if "emission" in data:
                material.emission = Vector3(*data["emission"])
            else:
                material.emission = Vector3(0, 0, 0)
            
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
        """Get number of interactive objects"""
        return len(self.scene.spheres) - 1
    
    def get_selected_object(self) -> Optional[Sphere]:
        """Get currently selected object"""
        idx = self.settings['selected_object']
        if 0 <= idx < len(self.scene.spheres):
            return self.scene.spheres[idx]
        return None
    
    def select_object_by_click(self, x: float, y: float) -> bool:
        """Select object by screen coordinates"""
        try:
            with self.render_lock:
                object_id = self.ray_tracer.select_object(x, y, self.width, self.height)
                if 0 <= object_id < len(self.scene.spheres):
                    self.settings['selected_object'] = object_id
                    obj = self.get_selected_object()
                    if obj:
                        return True
        except Exception as e:
            print(f"Object selection error: {e}")
        return False
    
    def start_object_dragging(self, x: float, y: float) -> bool:
        """Start dragging an object"""
        if self.select_object_by_click(x, y):
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:  # Don't drag ground
                self.dragging_object = True
                self.selected_object_id = obj.object_id
                self.drag_start_pos = (x, y)
                self.drag_start_object_pos = obj.center
                
                if self.render_mode == RenderMode.RAYTRACING:
                    self.render_mode = RenderMode.SILHOUETTE
                
                return True
        return False
    
    def update_object_dragging(self, dx: float, dy: float):
        """Update object position during dragging"""
        if not self.dragging_object:
            return
        
        obj = self.get_selected_object()
        if not obj or obj.object_id != self.selected_object_id:
            return
        
        speed = self.settings['move_speed'] * 2.0
        
        # Convert screen movement to world movement
        world_dx = self.camera_right * dx * 2.0
        world_dy = self.camera_up * (-dy) * 2.0
        
        # Apply dimension locks
        if self.lock_x:
            world_dx.x = 0
            world_dy.x = 0
        if self.lock_y:
            world_dx.y = 0
            world_dy.y = 0
        if self.lock_z:
            world_dx.z = 0
            world_dy.z = 0
        
        # Calculate new position
        move_vector = (world_dx + world_dy) * speed
        new_pos = self.drag_start_object_pos + move_vector
        
        # Apply bounds
        new_pos.x = max(-8, min(8, new_pos.x))
        new_pos.y = max(0.1, min(8, new_pos.y))
        new_pos.z = max(-8, min(2, new_pos.z))
        
        # Update object
        obj.center = new_pos
        self.ray_tracer.set_scene(self.scene)
        
        self._process_frame_for_display(0.016)
    
    def stop_object_dragging(self):
        """Stop dragging object"""
        self.dragging_object = False
        self.lock_x = self.lock_y = self.lock_z = False
        self.render_mode = RenderMode.RAYTRACING
        self.restart_rendering()
    
    def set_dimension_lock(self, dimension: str, state: bool):
        """Lock/unlock a dimension for dragging"""
        if dimension == 'x':
            self.lock_x = state
        elif dimension == 'y':
            self.lock_y = state
        elif dimension == 'z':
            self.lock_z = state
    
    def move_object(self, dx: float, dy: float, dz: float):
        """Move selected object with keyboard"""
        with self.render_lock:
            obj = self.get_selected_object()
            if obj and obj.object_id > 0:
                speed = self.settings['move_speed']
                obj.center.x += dx * speed
                obj.center.y += dy * speed
                obj.center.z += dz * speed
                
                # Bounds
                obj.center.x = max(-8, min(8, obj.center.x))
                obj.center.y = max(0.1, min(8, obj.center.y))
                obj.center.z = max(-8, min(2, obj.center.z))
                
                self.ray_tracer.set_scene(self.scene)
                self.restart_rendering()
    
    def update_object_material(self, property_name: str, value: float):
        """Update material property of selected object"""
        obj = self.get_selected_object()
        if obj:
            if property_name == 'albedo':
                obj.material.albedo = Vector3(value, value, value)
            elif property_name == 'metallic':
                obj.material.metallic = value
            elif property_name == 'roughness':
                obj.material.roughness = value
            
            self.restart_rendering()
    
    def update_light_intensity(self, intensity: float):
        """Update light intensity for selected light"""
        obj = self.get_selected_object()
        if obj and hasattr(obj.material, 'emission'):
            emission = obj.material.emission
            is_light = emission.x > 1.0 or emission.y > 1.0 or emission.z > 1.0
            
            if is_light:
                current_max = max(emission.x, emission.y, emission.z)
                if current_max > 0:
                    scale = intensity / current_max
                    obj.material.emission = Vector3(
                        emission.x * scale,
                        emission.y * scale,
                        emission.z * scale
                    )
                    self.restart_rendering()
    
    def restart_rendering(self):
        """Restart ray tracing"""
        with self.render_lock:
            self.is_rendering = False
            time.sleep(0.02)
            
            self.accumulated_image = None
            self.total_samples = 0
            self.frame_queue = Queue()
            
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
        """Worker function for ray tracing"""
        try:
            while (self.is_rendering and 
                   self.total_samples < self.settings['max_samples']):
                
                start_time = time.time()
                
                with self.render_lock:
                    result = self.ray_tracer.render(
                        self.width, self.height,
                        self.settings['samples_per_batch'],
                        self.settings['max_depth']
                    )
                
                if result is None or len(result) == 0:
                    continue
                
                # Process batch
                batch_image = np.array(result, dtype=np.float32).reshape(
                    (self.height, self.width, 3)
                )
                render_time = time.time() - start_time
                
                batch_samples = self.settings['samples_per_batch']
                
                if self.total_samples == 0:
                    self.accumulated_image = batch_image
                    self.total_samples = batch_samples
                else:
                    total_old = self.total_samples
                    total_new = self.total_samples + batch_samples
                    
                    weight_old = total_old / total_new
                    weight_new = batch_samples / total_new
                    
                    self.accumulated_image = (
                        self.accumulated_image * weight_old +
                        batch_image * weight_new
                    )
                    self.total_samples = total_new
                
                # Send frame if needed
                if (self.total_samples % self.settings['samples_per_batch'] == 0 or
                    self.total_samples >= self.settings['max_samples']):
                    self._process_frame_for_display(render_time)
                
                time.sleep(0.005)
                
        except Exception as e:
            print(f"Rendering error: {e}")
            import traceback
            traceback.print_exc()
        
        self.frame_queue.put({'done': True})
        self.is_rendering = False
    
    def _render_silhouette(self) -> np.ndarray:
        """Render silhouette view for fast object editing"""
        self.silhouette_buffer.fill(0)
        width, height = self.width, self.height
        
        # Camera parameters
        fov = self.camera.fov * 3.14159 / 180.0
        aspect_ratio = width / height
        tan_fov = np.tan(fov / 2.0)
        
        forward = (self.camera.target - self.camera.position).normalize()
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        up = right.cross(forward).normalize()
        
        for sphere in self.scene.spheres:
            if sphere.object_id == 0:  # Skip ground
                continue
            
            # Transform to camera space
            obj_pos = sphere.center - self.camera.position
            z_cam = obj_pos.dot(forward)
            
            if z_cam <= 0.1:  # Behind or too close to camera
                continue
            
            x_cam = obj_pos.dot(right)
            y_cam = obj_pos.dot(up)
            
            # Correct perspective projection (matches ray tracer)
            x_screen = (x_cam / (z_cam * tan_fov * aspect_ratio) + 0.5) * width
            # Fix Y axis inversion to match ray tracer
            y_screen = (0.5 - y_cam / (z_cam * tan_fov)) * height
            
            # Calculate projected radius
            radius_screen = (sphere.radius / (z_cam * tan_fov)) * height / 2.0
            
            # Clamp to screen bounds
            x_screen = max(0, min(width - 1, x_screen))
            y_screen = max(0, min(height - 1, y_screen))
            
            if 0 <= x_screen < width and 0 <= y_screen < height:
                center = (int(x_screen), int(y_screen))
                radius = max(2, int(radius_screen))
                
                # Color coding
                if sphere.object_id == self.selected_object_id:
                    color = (255, 255, 0)  # Yellow for selected
                    thickness = 3
                else:
                    color = (200, 200, 200)  # Gray for others
                    thickness = 1
                
                cv2.circle(self.silhouette_buffer, center, radius, color, thickness)
                
                # Crosshair for selected
                if sphere.object_id == self.selected_object_id:
                    cv2.line(self.silhouette_buffer,
                            (center[0] - 10, center[1]),
                            (center[0] + 10, center[1]),
                            (0, 255, 255), 2)
                    cv2.line(self.silhouette_buffer,
                            (center[0], center[1] - 10),
                            (center[0], center[1] + 10),
                            (0, 255, 255), 2)
        
        return self.silhouette_buffer.astype(np.float32) / 255.0
    
    def _render_wireframe(self) -> np.ndarray:
        """Render wireframe view for fast camera navigation"""
        self.wireframe_buffer.fill(0)
        width, height = self.width, self.height
        
        # Camera parameters
        fov = self.camera.fov * 3.14159 / 180.0
        aspect_ratio = width / height
        tan_fov = np.tan(fov / 2.0)
        
        forward = (self.camera.target - self.camera.position).normalize()
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        up = right.cross(forward).normalize()
        
        # Helper function with corrected Y axis
        def project_point(point: Vector3) -> Optional[Tuple[int, int]]:
            obj_pos = point - self.camera.position
            z_cam = obj_pos.dot(forward)
            
            if z_cam <= 0.1:
                return None
            
            x_cam = obj_pos.dot(right)
            y_cam = obj_pos.dot(up)
            
            # Correct projection with Y inversion
            x_screen = (x_cam / (z_cam * tan_fov * aspect_ratio) + 0.5) * width
            y_screen = (0.5 - y_cam / (z_cam * tan_fov)) * height
            
            # Clamp to screen bounds
            x_screen = max(0, min(width - 1, x_screen))
            y_screen = max(0, min(height - 1, y_screen))
            
            return (int(x_screen), int(y_screen))
        
        # Draw ground grid
        grid_size = 10
        grid_step = 1.0
        for i in range(-grid_size, grid_size + 1):
            x = i * grid_step
            
            # X lines
            for j in range(-grid_size, grid_size):
                z1 = j * grid_step
                z2 = (j + 1) * grid_step
                
                p1 = Vector3(x, 0, z1)
                p2 = Vector3(x, 0, z2)
                
                s1 = project_point(p1)
                s2 = project_point(p2)
                
                if s1 and s2:
                    cv2.line(self.wireframe_buffer, s1, s2, (80, 80, 80), 1)
            
            # Z lines
            for j in range(-grid_size, grid_size):
                x1 = j * grid_step
                x2 = (j + 1) * grid_step
                
                p1 = Vector3(x1, 0, x)
                p2 = Vector3(x2, 0, x)
                
                s1 = project_point(p1)
                s2 = project_point(p2)
                
                if s1 and s2:
                    cv2.line(self.wireframe_buffer, s1, s2, (80, 80, 80), 1)
        
        # Draw spheres
        for sphere in self.scene.spheres:
            if sphere.object_id == 0:
                continue
            
            center_screen = project_point(sphere.center)
            if center_screen:
                # Calculate screen radius
                distance = (sphere.center - self.camera.position).dot(forward)
                if distance > 0:
                    radius_screen = (sphere.radius / (distance * tan_fov)) * height / 2.0
                    radius_screen = max(2, int(radius_screen))
                    
                    # Color
                    if sphere.object_id == self.selected_object_id:
                        color = (255, 255, 0)
                        thickness = 2
                    else:
                        color = (200, 200, 200)
                        thickness = 1
                    
                    cv2.circle(self.wireframe_buffer, center_screen, radius_screen, color, thickness)
                    
                    # Axes for selected
                    if sphere.object_id == self.selected_object_id:
                        axes = [
                            (Vector3(0.5, 0, 0), (255, 0, 0)),   # X - Red
                            (Vector3(0, 0.5, 0), (0, 255, 0)),   # Y - Green
                            (Vector3(0, 0, -0.5), (0, 0, 255))   # Z - Blue
                        ]
                        
                        for axis_vec, axis_color in axes:
                            end = sphere.center + axis_vec
                            end_screen = project_point(end)
                            if end_screen:
                                cv2.line(self.wireframe_buffer, center_screen, end_screen, axis_color, 2)
        
        return self.wireframe_buffer.astype(np.float32) / 255.0
    
    def _process_frame_for_display(self, render_time: float):
        """Process frame for display based on current mode"""
        if self.render_mode == RenderMode.SILHOUETTE:
            display_image = self._render_silhouette()
            enhanced_image = display_image
            mode_str = "silhouette"
            denoised_images = {}
            
        elif self.render_mode == RenderMode.WIREFRAME:
            display_image = self._render_wireframe()
            enhanced_image = display_image
            mode_str = "wireframe"
            denoised_images = {}
            
        else:  # RAYTRACING
            if self.accumulated_image is None:
                return
            
            display_image = self._tone_map(self.accumulated_image, self.settings['exposure'])
            enhanced_image = self._enhance_display(display_image) if self.settings['enhance_image'] else display_image
            mode_str = "raytracing"
            
            # Apply denoisers if needed
            denoised_images = {}
            if self.settings['show_denoisers'] and self.settings['selected_denoisers']:
                for method in self.settings['selected_denoisers']:
                    try:
                        denoised_images[method] = self.denoiser.denoise(display_image, method)
                    except Exception as e:
                        print(f"Denoising error: {e}")
        
        frame_data = {
            'display': display_image,
            'enhanced': enhanced_image,
            'denoised': denoised_images,
            'samples': self.total_samples,
            'render_time': render_time,
            'mode': mode_str,
            'is_raytracing': self.render_mode == RenderMode.RAYTRACING
        }
        
        self.frame_queue.put(frame_data)
    
    def _tone_map(self, image: np.ndarray, exposure: float) -> np.ndarray:
        """Apply tone mapping"""
        image = image * exposure
        image = image / (1.0 + image)
        return np.clip(image, 0.0, 1.0)
    
    def _enhance_display(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast"""
        min_val = np.percentile(image, 2)
        max_val = np.percentile(image, 98)
        
        if max_val > min_val:
            enhanced = (image - min_val) / (max_val - min_val)
            return np.clip(enhanced, 0, 1)
        return image
    
    def has_frames(self) -> bool:
        """Check if frames are available"""
        return not self.frame_queue.empty()
    
    def get_frame(self) -> Optional[Dict]:
        """Get next frame"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
    
    def stop_rendering(self):
        """Stop all rendering"""
        self.is_rendering = False
        self.camera_move_active = False
        if self.camera_move_thread:
            self.camera_move_thread.join(timeout=1.0)