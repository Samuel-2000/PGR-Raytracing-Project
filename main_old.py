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

from denoisers.denoisers import Denoiser
from gui.inter import InteractiveGUI

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








def main():
    """Main entry point"""
    gui = InteractiveGUI()
    gui.run()

if __name__ == "__main__":
    main()