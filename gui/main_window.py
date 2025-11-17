# FILE: src/gui/main_window.py
"""
Modern GUI for Ray Tracer using PyQt
"""
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSplitter, QTabWidget, QGroupBox,
                            QLabel, QPushButton, QSlider, QComboBox, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter

from ..core.raytracer import RayTracer
from ..core.scene import Scene
from .render_view import RenderView
from .controls import ControlPanel

class RenderThread(QThread):
    """Separate thread for rendering"""
    frame_ready = pyqtSignal(np.ndarray, int)
    finished = pyqtSignal()
    
    def __init__(self, raytracer, width, height):
        super().__init__()
        self.raytracer = raytracer
        self.width = width
        self.height = height
        self.is_rendering = False
        self.samples_per_update = 4
    
    def run(self):
        """Main rendering loop"""
        self.is_rendering = True
        sample_count = 0
        
        while self.is_rendering and sample_count < self.raytracer.config['max_samples']:
            # Render batch
            self.raytracer.render_batch(self.samples_per_update)
            sample_count += self.samples_per_update
            
            # Get result and emit
            result = self.raytracer.get_progressive_result()
            self.frame_ready.emit(result, sample_count)
        
        self.finished.emit()
    
    def stop(self):
        """Stop rendering"""
        self.is_rendering = False

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.raytracer = RayTracer(800, 600)
        self.scene = self._create_default_scene()
        self.raytracer.set_scene(self.scene)
        
        self.render_thread = None
        self.setup_ui()
        self.setup_signals()
    
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("GPU Accelerated Ray Tracer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Render view
        self.render_view = RenderView()
        splitter.addWidget(self.render_view)
        
        # Control panel
        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.update_status("Ready")
    
    def setup_signals(self):
        """Connect signals and slots"""
        self.control_panel.render_button.clicked.connect(self.toggle_rendering)
        self.control_panel.settings_changed.connect(self.on_settings_changed)
    
    def _create_default_scene(self):
        """Create a default scene for demonstration"""
        scene = Scene()
        
        # Add some objects
        from ..core.materials import PBRMaterial
        
        # Ground
        ground_mat = PBRMaterial(
            albedo=[0.8, 0.8, 0.8],
            metallic=0.0,
            roughness=0.9
        )
        scene.add_sphere([0, -100.5, 0], 100, ground_mat)
        
        # Various spheres
        materials = [
            PBRMaterial([0.8, 0.2, 0.2], 0.9, 0.1),  # Red metallic
            PBRMaterial([0.2, 0.8, 0.2], 0.0, 0.3),  # Green dielectric
            PBRMaterial([0.2, 0.2, 0.8], 0.0, 0.0),  # Blue glass
            PBRMaterial([0.8, 0.8, 0.2], 0.5, 0.2),  # Yellow mixed
        ]
        
        positions = [
            [-2.0, 0.0, -3.0],
            [0.0, 0.0, -3.0],
            [2.0, 0.0, -3.0],
            [0.0, 1.0, -2.0],
        ]
        
        for pos, mat in zip(positions, materials):
            scene.add_sphere(pos, 0.5, mat)
        
        # Light
        light_mat = PBRMaterial(
            albedo=[1, 1, 1],
            emission=[10, 10, 8]
        )
        scene.add_sphere([0, 5, 0], 1.0, light_mat)
        
        return scene
    
    def toggle_rendering(self):
        """Start or stop rendering"""
        if self.render_thread and self.render_thread.is_rendering:
            self.stop_rendering()
        else:
            self.start_rendering()
    
    def start_rendering(self):
        """Start rendering in separate thread"""
        if self.render_thread:
            self.render_thread.wait()
        
        self.render_thread = RenderThread(self.raytracer, 800, 600)
        self.render_thread.frame_ready.connect(self.on_frame_ready)
        self.render_thread.finished.connect(self.on_render_finished)
        
        self.control_panel.render_button.setText("Stop Rendering")
        self.update_status("Rendering...")
        
        self.render_thread.start()
    
    def stop_rendering(self):
        """Stop rendering"""
        if self.render_thread:
            self.render_thread.stop()
            self.render_thread.wait()
        
        self.control_panel.render_button.setText("Start Rendering")
        self.update_status("Stopped")
    
    def on_frame_ready(self, image: np.ndarray, samples: int):
        """Update display with new frame"""
        self.render_view.update_image(image)
        self.progress_bar.setValue(
            min(100, int(100 * samples / self.raytracer.config['max_samples']))
        )
        self.update_status(f"Rendering: {samples} samples")
    
    def on_render_finished(self):
        """Handle render completion"""
        self.control_panel.render_button.setText("Start Rendering")
        self.update_status("Render Complete")
    
    def on_settings_changed(self, setting: str, value):
        """Handle settings changes"""
        if setting in self.raytracer.config:
            self.raytracer.config[setting] = value
        
        if setting in ['max_samples', 'max_bounces']:
            self.raytracer.reset_accumulator()
            if self.render_thread and self.render_thread.is_rendering:
                self.stop_rendering()
                self.start_rendering()
    
    def update_status(self, message: str):
        """Update status bar"""
        self.status_bar.showMessage(message)
    
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_rendering()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = MainWindow()
    window.show()
    
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())