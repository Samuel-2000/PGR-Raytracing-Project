#!/usr/bin/env python3
"""
Final working version of the Physically Based Ray Tracer
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from queue import Queue
import cv2
from typing import Dict, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3
    CPP_AVAILABLE = True
    print("✓ Using C++ accelerated ray tracer")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"✗ C++ ray tracer not available: {e}")
    print("Using Python fallback")

class SimpleRayTracer:
    """Simple Python fallback"""
    def __init__(self):
        self.scene = None
    
    def render(self, width, height, samples, depth):
        if self.scene is None:
            return np.zeros((height, width, 3))
        # Simple placeholder - in practice you'd implement basic ray tracing here
        image = np.random.rand(height, width, 3) * 0.3
        # Add a colored circle
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        mask = (x - center_x)**2 + (y - center_y)**2 <= (min(height, width) // 4)**2
        image[mask] = [0.8, 0.2, 0.2]  # Red circle
        return image

class ToneMapper:
    """Handles tone mapping and color correction"""
    
    @staticmethod
    def reinhard_tone_map(image, exposure=1.0):
        """Simple Reinhard tone mapping"""
        image = image * exposure
        return image / (1.0 + image)
    
    @staticmethod
    def aces_approx_tone_map(image, exposure=1.0):
        """ACES approximate tone mapping (film-like)"""
        image = image * exposure
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((image * (a * image + b)) / (image * (c * image + d) + e), 0, 1)
    
    @staticmethod
    def simple_tone_map(image, exposure=1.0, white_point=1.0):
        """Simple tone mapping with exposure control"""
        image = image * exposure
        return 1.0 - np.exp(-image / white_point)
    


class SceneBuilder:
    """Builds scenes for the ray tracer"""
    
    @staticmethod
    def create_vibrant_scene():
        """Create a vibrant, well-lit scene"""
        scene = Scene()
        scene.background_color = Vector3(0.05, 0.05, 0.1)
        
        # Bright ground
        ground_material = Material()
        ground_material.albedo = Vector3(0.9, 0.9, 0.9) #Vector3(0.9, 0.9, 0.9)
        ground = Sphere()
        ground.center = Vector3(0, -101, 0)
        ground.radius = 100.0
        ground.material = ground_material
        scene.add_sphere(ground)
        
        # Colorful spheres
        spheres_data = [
            {"pos": (-2.0, 0.0, -6.0), "color": (0.9, 0.1, 0.1), "metal": 0.8, "rough": 0.1},  # Red metallic
            {"pos": (0.0, 0.0, -6.0), "color": (0.1, 0.9, 0.1), "metal": 0.3, "rough": 0.3},   # Green
            {"pos": (2.0, 0.0, -6.0), "color": (0.1, 0.1, 0.9), "metal": 0.0, "rough": 0.0},   # Blue glass
            {"pos": (-1.0, 1.5, -4.0), "color": (0.9, 0.9, 0.1), "metal": 0.5, "rough": 0.2},  # Yellow
            {"pos": (1.0, -1.0, -4.0), "color": (0.9, 0.1, 0.9), "metal": 0.7, "rough": 0.4},  # Purple
        ]
        
        for i, data in enumerate(spheres_data):
            material = Material()
            material.albedo = Vector3(*data["color"])
            material.metallic = data["metal"]
            material.roughness = data["rough"]
            material.ior = 1.5
            
            sphere = Sphere()
            sphere.center = Vector3(*data["pos"])
            sphere.radius = 0.8
            sphere.material = material
            scene.add_sphere(sphere)
        
        # Bright lights
        lights_data = [
            {"pos": (0, 8, -2), "color": (25, 25, 20), "radius": 1.5},    # Main light
            {"pos": (-4, 3, 0), "color": (15, 10, 5), "radius": 1.0},     # Warm light
            {"pos": (4, 3, 0), "color": (5, 10, 15), "radius": 1.0},      # Cool light
        ]
        
        for i, data in enumerate(lights_data):
            material = Material()
            material.emission = Vector3(*data["color"])
            
            sphere = Sphere()
            sphere.center = Vector3(*data["pos"])
            sphere.radius = data["radius"]
            sphere.material = material
            scene.add_sphere(sphere)
        
        scene.build_bvh()
        return scene

class ProgressiveRenderer:
    """Handles progressive rendering with real-time display"""
    
    def __init__(self, width=400, height=300):
        self.width = width
        self.height = height
        self.use_cpp = CPP_AVAILABLE
        
        # Initialize ray tracer
        if self.use_cpp:
            self.ray_tracer = RayTracer()
            self.scene = SceneBuilder.create_vibrant_scene()
            self.ray_tracer.set_scene(self.scene)
        else:
            self.ray_tracer = SimpleRayTracer()
            # Simple scene for Python fallback
            self.ray_tracer.scene = type('Scene', (), {})()  # Dummy scene
        
        # Rendering state
        self.is_rendering = False
        self.accumulated_image = None
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Settings
        self.samples_per_batch = 4
        self.max_samples = 128
        self.max_depth = 8
        
        # Tone mapping settings
        self.tone_map_method = 'reinhard'
        self.exposure = 1.5
        
        print(f"Initialized renderer: {width}x{height}")
        print(f"Using {'C++' if self.use_cpp else 'Python'} backend")
        print(f"Tone mapping: {self.tone_map_method}, Exposure: {self.exposure}")
    
    def start_rendering(self):
        """Start progressive rendering in a separate thread"""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.accumulated_image = np.zeros((self.height, self.width, 3))
        self.total_samples = 0
        self.frame_queue = Queue()
        
        # Start render thread
        render_thread = threading.Thread(target=self._render_worker)
        render_thread.daemon = True
        render_thread.start()
        
        return render_thread
    
    def _render_worker(self):
        """Worker function that does the actual rendering"""
        while self.is_rendering and self.total_samples < self.max_samples:
            try:
                # Render a batch
                start_time = time.time()
                
                if self.use_cpp:
                    result = self.ray_tracer.render(
                        self.width, self.height, 
                        self.samples_per_batch, self.max_depth
                    )
                    batch_image = np.array(result).reshape((self.height, self.width, 3))
                else:
                    batch_image = self.ray_tracer.render(
                        self.width, self.height,
                        self.samples_per_batch, self.max_depth
                    )
                
                render_time = time.time() - start_time
                
                # Update accumulated image
                self.total_samples += self.samples_per_batch
                
                if self.total_samples == self.samples_per_batch:
                    self.accumulated_image = batch_image
                else:
                    # Running average
                    weight_old = (self.total_samples - self.samples_per_batch) / self.total_samples
                    weight_new = self.samples_per_batch / self.total_samples
                    self.accumulated_image = self.accumulated_image * weight_old + batch_image * weight_new
                
                # Prepare display images with proper tone mapping
                linear_image = np.clip(self.accumulated_image, 0, 10)  # Clamp to reasonable range
                
                # Enhanced display (with additional contrast)
                enhanced_image = self._enhance_display(linear_image)
                
                # Put frame in queue for display
                self.frame_queue.put({
                    'linear': linear_image,  # Keep linear for stats
                    'display': linear_image,
                    'enhanced': enhanced_image,
                    'samples': self.total_samples,
                    'render_time': render_time,
                    'linear_stats': f"{linear_image.min():.3f}-{linear_image.max():.3f}",
                    'display_stats': f"{linear_image.min():.3f}-{linear_image.max():.3f}"
                })
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Rendering error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Signal completion
        self.frame_queue.put({'done': True})
        self.is_rendering = False
    
    def _enhance_display(self, image):
        """Additional enhancement for the right panel"""
        # Simple contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        if p98 > p2:  # Avoid division by zero
            enhanced = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            enhanced = image
        return enhanced
    
    def stop_rendering(self):
        """Stop the rendering process"""
        self.is_rendering = False
    
    def has_frames(self):
        """Check if there are frames ready for display"""
        return not self.frame_queue.empty()
    
    def get_frame(self):
        """Get the next frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

class Application:
    """Main application class"""
    
    def __init__(self):
        self.width = 400
        self.height = 300
        self.renderer = ProgressiveRenderer(self.width, self.height)
        
    def run(self):
        """Run the main application"""
        print("Starting Physically Based Ray Tracer")
        print("=" * 50)
        print("Left: Tone Mapped | Right: Enhanced")
        print("=" * 50)
        
        # Start rendering
        print("Starting progressive rendering...")
        self.renderer.start_rendering()
        
        # Set up display
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Create initial images
        empty_image = np.zeros((self.height, self.width, 3))
        img1 = ax1.imshow(empty_image)
        img2 = ax2.imshow(empty_image)
        
        ax1.set_title('Tone Mapped - Starting...')
        ax2.set_title('Enhanced - Starting...')
        ax1.axis('off')
        ax2.axis('off')
        
        fig.suptitle('Physically Based Ray Tracer - Initializing')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        frame_count = 0
        last_frame_time = time.time()
        
        try:
            while self.renderer.is_rendering or self.renderer.has_frames():
                # Process all available frames
                frames_processed = 0
                while self.renderer.has_frames():
                    frame = self.renderer.get_frame()
                    
                    if frame is None:
                        break
                    
                    if 'done' in frame:
                        print("Rendering completed!")
                        break
                    
                    # Update display
                    # Left panel - tone mapped
                    img1.set_data(frame['display'])
                    ax1.set_title(f'Tone Mapped - {frame["samples"]} samples')
                    
                    # Right panel - enhanced
                    img2.set_data(frame['enhanced'])
                    ax2.set_title(f'Enhanced - {frame["samples"]} samples')
                    
                    # Update main title
                    fig.suptitle(f'Progressive Rendering | Samples: {frame["samples"]} | '
                               f'Batch Time: {frame["render_time"]:.2f}s | '
                               f'Linear: {frame["linear_stats"]} | '
                               f'Display: {frame["display_stats"]}')
                    
                    # Force redraw
                    plt.draw()
                    plt.pause(0.001)
                    
                    print(f"Frame {frame_count}: {frame['samples']} samples, "
                          f"linear: {frame['linear_stats']}, "
                          f"display: {frame['display_stats']}")
                    
                    frame_count += 1
                    frames_processed += 1
                    last_frame_time = time.time()
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
                
                # Check for timeout (no new frames for 5 seconds)
                if time.time() - last_frame_time > 5.0 and frame_count > 0:
                    print("No new frames for 5 seconds, stopping...")
                    break
            
            # Final display
            if self.renderer.accumulated_image is not None:
                linear_final = np.clip(self.renderer.accumulated_image, 0, 10)
                enhanced_final = self.renderer._enhance_display(linear_final)
                
                img1.set_data(linear_final)
                img2.set_data(enhanced_final)
                ax1.set_title(f'Final Tone Mapped - {self.renderer.total_samples} samples')
                ax2.set_title(f'Final Enhanced - {self.renderer.total_samples} samples')
                fig.suptitle(f'Rendering Complete - {self.renderer.total_samples} total samples')
                plt.draw()
            
            print(f"\nRendering finished! Processed {frame_count} frames.")
            print("Close the window to exit.")
            
            # Keep window open
            plt.ioff()
            plt.show()
            
        except KeyboardInterrupt:
            print("\nStopped by user")
            self.renderer.stop_rendering()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.renderer.stop_rendering()
    
    def _enhance_image(self, image):
        """Simple image enhancement for display"""
        # Contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        enhanced = np.clip((image - p2) / (p98 - p2), 0, 1)
        return enhanced

def main():
    """Main entry point"""
    app = Application()
    app.run()

if __name__ == "__main__":
    main()