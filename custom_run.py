#!/usr/bin/env python3
"""
Custom runner with different settings
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Application, ProgressiveRenderer

class CustomApplication(Application):
    def __init__(self):
        # Higher quality settings
        self.width = 512
        self.height = 384
        self.renderer = ProgressiveRenderer(self.width, self.height)
        # Override renderer settings
        self.renderer.samples_per_batch = 2  # More frequent updates
        self.renderer.max_samples = 256     # Higher quality
        self.renderer.max_depth = 12        # More reflections

if __name__ == "__main__":
    app = CustomApplication()
    app.run()