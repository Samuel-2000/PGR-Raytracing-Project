#!/usr/bin/env python3
"""
Simple runner for the ray tracer
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import main
    print("Starting Physically Based Ray Tracer...")
    print("Features: BVH Acceleration, PBR Materials, Progressive Rendering")
    main()
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
except KeyboardInterrupt:
    print("\nRay tracer stopped by user")
except Exception as e:
    print(f"Unexpected error: {e}")