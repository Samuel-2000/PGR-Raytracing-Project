#!/usr/bin/env python3
"""
Enhanced Interactive Ray Tracer - C++ Only Version
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import time
import threading
from queue import Queue
import cv2
from typing import Dict, Any, List, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import C++ ray tracer - this will fail if not built
try:
    from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3
    CPP_AVAILABLE = True
    print("✓ Using C++ accelerated ray tracer")
except ImportError as e:
    print(f"❌ C++ ray tracer not available: {e}")
    print("Please build the C++ extension first:")
    print("cd cpp_raytracer && python setup.py build_ext --inplace")
    sys.exit(1)

class Denoiser:
    """Denoising algorithms"""
    
    def __init__(self):
        self.available_methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
    
    def denoise(self, image: np.ndarray, method: str = 'bilateral', **kwargs) -> np.ndarray:
        """Apply denoising to image"""
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        if method == 'bilateral':
            return self._bilateral_denoise(image_uint8, **kwargs)
        elif method == 'nlmeans':
            return self._nlmeans_denoise(image_uint8, **kwargs)
        elif method == 'gaussian':
            return self._gaussian_denoise(image_uint8, **kwargs)
        elif method == 'median':
            return self._median_denoise(image_uint8, **kwargs)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def _bilateral_denoise(self, image: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised.astype(np.float32) / 255.0
    
    def _nlmeans_denoise(self, image: np.ndarray, h: float = 10,
                        template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        )
        return denoised.astype(np.float32) / 255.0
    
    def _gaussian_denoise(self, image: np.ndarray, kernel_size: int = 5,
                         sigma: float = 1.0) -> np.ndarray:
        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return denoised.astype(np.float32) / 255.0
    
    def _median_denoise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        denoised = cv2.medianBlur(image, kernel_size)
        return denoised.astype(np.float32) / 255.0

class InteractiveRayTracer:
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

class InteractiveGUI:
    """Interactive GUI for the ray tracer"""
    
    def __init__(self):
        self.raytracer = InteractiveRayTracer(400, 300)
        self.fig = None
        self.axes = {}
        self.images = {}
        self.controls = {}
        self.key_states = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('C++ Ray Tracer - Interactive Controls', fontsize=14, fontweight='bold')
        
        # Main image display area
        self.axes['main'] = plt.axes([0.02, 0.25, 0.6, 0.7])
        self.axes['main'].set_title('Main View - Tone Mapped')
        self.axes['main'].axis('off')
        
        # Enhanced view
        self.axes['enhanced'] = plt.axes([0.63, 0.55, 0.35, 0.4])
        self.axes['enhanced'].set_title('Enhanced View')
        self.axes['enhanced'].axis('off')
        
        # Create initial images
        empty = np.zeros((self.raytracer.height, self.raytracer.width, 3))
        self.images['main'] = self.axes['main'].imshow(empty)
        self.images['enhanced'] = self.axes['enhanced'].imshow(empty)
        
        self.setup_controls()
        self.setup_event_handlers()
    
    def setup_controls(self):
        """Setup all control elements"""
        # Rendering settings
        self.setup_rendering_controls()
        
        # Object controls
        self.setup_object_controls()
        
        # Material controls  
        self.setup_material_controls()
        
        # Denoiser controls
        self.setup_denoiser_controls()
        
        # Info display
        self.setup_info_display()
    
    def setup_rendering_controls(self):
        """Setup rendering control sliders"""
        y_pos = 0.18
        slider_width = 0.25
        
        # Max Samples
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['max_samples'] = Slider(ax, 'Max Samples', 1, 512, valinit=128, valstep=1)
        self.controls['max_samples'].on_changed(self.on_max_samples_changed)
        y_pos -= 0.03
        
        # Samples per Batch
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['samples_per_batch'] = Slider(ax, 'Samples/Batch', 1, 32, valinit=4, valstep=1)
        self.controls['samples_per_batch'].on_changed(self.on_samples_per_batch_changed)
        y_pos -= 0.03
        
        # Max Depth
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['max_depth'] = Slider(ax, 'Max Depth', 1, 16, valinit=8, valstep=1)
        self.controls['max_depth'].on_changed(self.on_max_depth_changed)
        y_pos -= 0.03
        
        # Update Step
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['update_step'] = Slider(ax, 'Update Step', 1, 32, valinit=4, valstep=1)
        self.controls['update_step'].on_changed(self.on_update_step_changed)
        y_pos -= 0.03
        
        # Exposure
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['exposure'] = Slider(ax, 'Exposure', 0.1, 5.0, valinit=1.5)
        self.controls['exposure'].on_changed(self.on_exposure_changed)
        y_pos -= 0.03
        
        # Move Speed
        ax = plt.axes([0.65, y_pos, slider_width, 0.02])
        self.controls['move_speed'] = Slider(ax, 'Move Speed', 0.1, 2.0, valinit=0.5)
        self.controls['move_speed'].on_changed(self.on_move_speed_changed)
    
    def setup_object_controls(self):
        """Setup object manipulation controls"""
        y_pos = 0.18
        slider_width = 0.25
        
        # Object selection
        ax = plt.axes([0.05, y_pos, slider_width, 0.02])
        object_count = self.raytracer.get_object_count()
        self.controls['object_select'] = Slider(
            ax, 'Object', 0, object_count - 1, valinit=0, valstep=1
        )
        self.controls['object_select'].on_changed(self.on_object_selected)
        y_pos -= 0.03
        
        # Object info
        self.axes['object_info'] = plt.axes([0.05, y_pos - 0.02, slider_width, 0.05])
        self.axes['object_info'].axis('off')
        self.axes['object_info'].text(0, 0.5, 'Selected: Red Metallic', fontsize=10, va='center')
    
    def setup_material_controls(self):
        """Setup material property controls"""
        y_pos = 0.08
        slider_width = 0.12
        
        # Metallic
        ax = plt.axes([0.05, y_pos, slider_width, 0.02])
        self.controls['metallic'] = Slider(ax, 'Metallic', 0, 1, valinit=0.9)
        self.controls['metallic'].on_changed(lambda x: self.on_material_changed('metallic', x))
        
        # Roughness
        ax = plt.axes([0.20, y_pos, slider_width, 0.02])
        self.controls['roughness'] = Slider(ax, 'Roughness', 0, 1, valinit=0.1)
        self.controls['roughness'].on_changed(lambda x: self.on_material_changed('roughness', x))
        y_pos -= 0.03
        
        # Color Red
        ax = plt.axes([0.05, y_pos, slider_width, 0.02])
        self.controls['color_r'] = Slider(ax, 'Color R', 0, 1, valinit=0.9)
        self.controls['color_r'].on_changed(lambda x: self.on_material_changed('albedo', x))
        
        # Light Intensity
        ax = plt.axes([0.20, y_pos, slider_width, 0.02])
        self.controls['light_intensity'] = Slider(ax, 'Light Power', 1, 50, valinit=15)
        self.controls['light_intensity'].on_changed(self.on_light_intensity_changed)
    
    def setup_denoiser_controls(self):
        """Setup denoiser controls"""
        # Checkbox for show denoisers
        ax = plt.axes([0.65, 0.05, 0.15, 0.03])
        self.controls['show_denoisers'] = CheckButtons(ax, ['Show Denoisers'], [False])
        self.controls['show_denoisers'].on_clicked(self.on_show_denoisers_changed)
        
        # Denoiser selection
        ax = plt.axes([0.65, 0.01, 0.25, 0.03])
        self.controls['denoiser_select'] = CheckButtons(
            ax, ['Bilateral', 'NL-Means', 'Gaussian', 'Median'], 
            [True, False, False, False]
        )
        self.controls['denoiser_select'].on_clicked(self.on_denoiser_selection_changed)
        
        # Denoiser display areas
        denoiser_positions = [
            [0.02, 0.55, 0.18, 0.18],  # Top-left
            [0.21, 0.55, 0.18, 0.18],  # Top-right  
            [0.02, 0.35, 0.18, 0.18],  # Bottom-left
            [0.21, 0.35, 0.18, 0.18],  # Bottom-right
        ]
        
        methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
        for i, (method, pos) in enumerate(zip(methods, denoiser_positions)):
            ax = plt.axes(pos)
            ax.set_title(f'{method.title()} Denoised')
            ax.axis('off')
            ax.set_visible(False)  # Start hidden
            self.axes[f'denoiser_{method}'] = ax
            self.images[f'denoiser_{method}'] = ax.imshow(np.zeros((150, 200, 3)))
    
    def setup_info_display(self):
        """Setup information display"""
        self.axes['info'] = plt.axes([0.65, 0.22, 0.3, 0.03])
        self.axes['info'].axis('off')
        self.info_text = self.axes['info'].text(0, 0.5, 'Ready to render...', fontsize=10, va='center')
    
    def setup_event_handlers(self):
        """Setup keyboard and mouse event handlers"""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
    
    def update_object_info(self):
        """Update object information display"""
        obj = self.raytracer.get_selected_object()
        if obj:
            text = f"Selected: {getattr(obj, 'name', 'Object')}"
            self.axes['object_info'].clear()
            self.axes['object_info'].axis('off')
            self.axes['object_info'].text(0, 0.5, text, fontsize=10, va='center')
            self.fig.canvas.draw_idle()
    
    def update_material_sliders(self):
        """Update material sliders to match selected object"""
        obj = self.raytracer.get_selected_object()
        if obj:
            mat = obj.material
            self.controls['metallic'].set_val(mat.metallic)
            self.controls['roughness'].set_val(mat.roughness)
            self.controls['color_r'].set_val(mat.albedo.x)
            
            # Update light intensity if it's a light
            if mat.emission.x > 0:
                self.controls['light_intensity'].set_val(mat.emission.x)
    
    # Event handlers
    def on_max_samples_changed(self, val):
        self.raytracer.settings['max_samples'] = int(val)
        self.raytracer.restart_rendering()
    
    def on_samples_per_batch_changed(self, val):
        self.raytracer.settings['samples_per_batch'] = int(val)
    
    def on_max_depth_changed(self, val):
        self.raytracer.settings['max_depth'] = int(val)
        self.raytracer.restart_rendering()
    
    def on_update_step_changed(self, val):
        self.raytracer.settings['update_step'] = int(val)
    
    def on_exposure_changed(self, val):
        self.raytracer.settings['exposure'] = val
    
    def on_move_speed_changed(self, val):
        self.raytracer.settings['move_speed'] = val
    
    def on_object_selected(self, val):
        self.raytracer.settings['selected_object'] = int(val)
        self.update_object_info()
        self.update_material_sliders()
    
    def on_material_changed(self, property_name, value):
        self.raytracer.update_object_material(property_name, value)
    
    def on_light_intensity_changed(self, value):
        self.raytracer.update_light_intensity(value)
    
    def on_show_denoisers_changed(self, label):
        self.raytracer.settings['show_denoisers'] = not self.raytracer.settings['show_denoisers']
        self.update_display_layout()
    
    def on_denoiser_selection_changed(self, label):
        labels = ['Bilateral', 'NL-Means', 'Gaussian', 'Median']
        methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
        
        selected = []
        for i, (l, m) in enumerate(zip(labels, methods)):
            if self.controls['denoiser_select'].get_status()[i]:
                selected.append(m)
        
        self.raytracer.settings['selected_denoisers'] = selected
    
    def on_key_press(self, event):
        """Handle keyboard input for object movement"""
        if event.key in ['left', 'right', 'up', 'down', 'w', 'a', 's', 'd', 'q', 'e']:
            self.key_states[event.key] = True
    
    def on_key_release(self, event):
        """Handle key release"""
        if event.key in self.key_states:
            self.key_states[event.key] = False
    
    def on_mouse_press(self, event):
        """Handle mouse press for object dragging"""
        if event.inaxes == self.axes['main'] and event.button == 1:
            self.raytracer.dragging = True
            self.raytracer.drag_start = (event.xdata, event.ydata)
    
    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.raytracer.dragging = False
        self.raytracer.drag_start = None
    
    def on_mouse_move(self, event):
        """Handle mouse movement for object dragging"""
        if self.raytracer.dragging and event.inaxes == self.axes['main']:
            if self.raytracer.drag_start:
                dx = (event.xdata - self.raytracer.drag_start[0]) / 50
                dy = (event.ydata - self.raytracer.drag_start[1]) / 50
                self.raytracer.move_object(dx, dy, 0)
                self.raytracer.drag_start = (event.xdata, event.ydata)
    
    def update_display_layout(self):
        """Update display layout based on settings"""
        show_denoisers = self.raytracer.settings['show_denoisers']
        
        # Show/hide denoiser panels
        for method in ['bilateral', 'nlmeans', 'gaussian', 'median']:
            ax = self.axes[f'denoiser_{method}']
            ax.set_visible(show_denoisers)
        
        self.fig.canvas.draw_idle()
    
    def process_keyboard_input(self):
        """Process continuous keyboard input for movement"""
        move_dict = {
            'left': (-1, 0, 0), 'right': (1, 0, 0),
            'up': (0, 1, 0), 'down': (0, -1, 0),
            'w': (0, 0, -1), 's': (0, 0, 1),
            'a': (-1, 0, 0), 'd': (1, 0, 0),
            'q': (0, 1, 0), 'e': (0, -1, 0)
        }
        
        moved = False
        for key, move in move_dict.items():
            if self.key_states.get(key, False):
                self.raytracer.move_object(*move)
                moved = True
                break
        
        return moved
    
    def run(self):
        """Main application loop"""
        print("Starting C++ Ray Tracer with Interactive GUI")
        print("=" * 60)
        print("Controls:")
        print("- WASD: Move object horizontally")
        print("- Q/E: Move object vertically") 
        print("- Arrow keys: Alternative movement")
        print("- Left click + drag: Drag object in view plane")
        print("- Sliders: Adjust rendering and material settings")
        print("- Checkboxes: Toggle denoisers and features")
        print("=" * 60)
        
        self.raytracer.start_rendering()
        self.update_object_info()
        self.update_material_sliders()
        
        try:
            while True:
                # Process keyboard input
                if self.process_keyboard_input():
                    pass
                
                # Process frames from renderer
                while self.raytracer.has_frames():
                    frame = self.raytracer.get_frame()
                    if frame is None:
                        break
                    
                    if 'done' in frame:
                        print("Rendering completed")
                        self.info_text.set_text("Rendering Complete!")
                        break
                    
                    # Update displays
                    self.images['main'].set_data(frame['display'])
                    self.images['enhanced'].set_data(frame['enhanced'])
                    
                    # Update denoiser displays if enabled
                    if self.raytracer.settings['show_denoisers']:
                        for method, image in frame.get('denoised', {}).items():
                            img_key = f'denoiser_{method}'
                            if img_key in self.images:
                                small_img = cv2.resize(image, (200, 150))
                                self.images[img_key].set_data(small_img)
                    
                    # Update info
                    info = (f"Samples: {frame['samples']} | "
                           f"Batch Time: {frame['render_time']:.3f}s | "
                           f"FPS: {1/frame['render_time']:.1f}" if frame['render_time'] > 0 else "Rendering...")
                    self.info_text.set_text(info)
                    
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                
                # Small delay to prevent busy waiting
                plt.pause(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping ray tracer...")
        finally:
            self.raytracer.stop_rendering()
            plt.close('all')

def main():
    """Main entry point"""
    gui = InteractiveGUI()
    gui.run()

if __name__ == "__main__":
    main()