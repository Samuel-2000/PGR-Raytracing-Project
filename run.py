#!/usr/bin/env python3
# run.py

import os
import sys
import subprocess
import platform

def build_cpp_extension():
    """Build the C++ ray tracer extension"""
    print("🔨 Building C++ Ray Tracer Extension...")
    
    # Check if we're in the right directory
    cpp_dir = "cpp_raytracer"
    if not os.path.exists(cpp_dir):
        print(f"❌ Error: {cpp_dir} directory not found!")
        print("Please run this script from the project root directory")
        return False
    
    # Change to C++ directory
    original_dir = os.getcwd()
    os.chdir(cpp_dir)
    
    try:
        # Build the extension
        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ C++ extension built successfully!")
            
            # List the built files
            built_files = [f for f in os.listdir('.') if f.endswith(('.so', '.pyd', '.cpp'))]
            print(f"📁 Built files: {', '.join(built_files)}")
            
            return True
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def check_cpp_extension():
    """Check if C++ extension is available"""
    try:
        from cpp_raytracer.raytracer_cpp import RayTracer, Scene, Sphere, Material, Vector3
        print("✅ C++ extension is available")
        return True
    except ImportError as e:
        print(f"❌ C++ extension not available: {e}")
        return False

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    if os.path.exists("requirements.txt"):
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print("❌ Failed to install requirements!")
            print("STDERR:", result.stderr)
            return False
    else:
        print("❌ requirements.txt not found!")
        return False

def main():
    """Main build and run function"""
    print("🚀 C++ Ray Tracer - Build and Run")
    print("=" * 50)
    
    # Check platform
    system = platform.system()
    print(f"💻 Platform: {system} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    
    # Install requirements first
    if not install_requirements():
        print("⚠️  Continuing with build...")
    
    # Check if C++ extension is already available
    if check_cpp_extension():
        print("🎯 Starting ray tracer...")
    else:
        # Build the C++ extension
        if not build_cpp_extension():
            print("💥 Failed to build C++ extension!")
            sys.exit(1)
        
        # Verify the build
        if not check_cpp_extension():
            print("💥 C++ extension still not available after build!")
            sys.exit(1)
    
    # Import and run the main application
    try:
        from PyQt5.QtWidgets import QApplication
        from gui import GUI

        app = QApplication(sys.argv)
        
        print("\n" + "=" * 50)
        print("🎮 Starting Interactive Ray Tracer GUI...")
        print("=" * 50)
        
        gui = GUI()
        gui.show()
        
        print("Controls:")
        print("- WASD/Arrow keys: Move selected object")
        print("- Q/E: Move object up/down") 
        print("- Left click + drag: Drag object in view plane")
        print("- UI controls: Adjust rendering and material settings")
        print("=" * 50)
        
        sys.exit(app.exec_())


    except ImportError as e:
        print(f"💥 Failed to import main application: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Ray tracer stopped by user")

    except Exception as e:
        print(f"💥 Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()