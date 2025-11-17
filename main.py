# FILE: main.py
"""
Main entry point for the optimized ray tracer
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point"""
    try:
        from gui.main_window import main as gui_main
        return gui_main()
    except ImportError as e:
        print(f"GUI dependencies not available: {e}")
        print("Falling back to console mode...")
        
        # Fallback to console rendering
        from core.raytracer import RayTracer
        from core.scene import Scene
        from core.materials import PBRMaterial
        import matplotlib.pyplot as plt
        
        # Create simple scene
        scene = Scene()
        
        # Add objects
        ground_mat = PBRMaterial([0.8, 0.8, 0.8], 0.0, 0.9)
        scene.add_sphere([0, -100.5, 0], 100, ground_mat)
        
        sphere_mat = PBRMaterial([0.8, 0.2, 0.2], 0.9, 0.1)
        scene.add_sphere([0, 0, -3], 0.5, sphere_mat)
        
        light_mat = PBRMaterial([1, 1, 1], 0.0, 0.1, [10, 10, 8])
        scene.add_sphere([0, 5, 0], 1.0, light_mat)
        
        # Render
        raytracer = RayTracer(400, 300)
        raytracer.set_scene(scene)
        
        print("Rendering...")
        for i in range(4):
            raytracer.render_batch(16)
            print(f"Completed {raytracer.sample_count} samples")
        
        result = raytracer.get_progressive_result()
        
        # Display
        plt.imshow(result)
        plt.title(f"Ray Traced Image ({raytracer.sample_count} samples)")
        plt.axis('off')
        plt.show()
        
        return 0

if __name__ == '__main__':
    sys.exit(main())