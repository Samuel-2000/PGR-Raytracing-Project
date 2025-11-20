# gui.py

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QSlider, QCheckBox, QComboBox, QLabel, QPushButton,
                             QTabWidget, QSplitter, QProgressBar, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
import cv2

from interaction import RayTracerInteraction


class RenderThread(QThread):
    """Thread for handling rendering updates"""
    frame_ready = pyqtSignal(dict)
    rendering_finished = pyqtSignal()
    
    def __init__(self, raytracer):
        super().__init__()
        self.raytracer = raytracer
        self.running = True
        
    def run(self):
        """Main rendering loop"""
        self.raytracer.start_rendering()
        
        while self.running:
            # Process frames from renderer
            while self.raytracer.has_frames():
                frame = self.raytracer.get_frame()
                if frame is None:
                    break
                
                if 'done' in frame:
                    self.rendering_finished.emit()
                    break
                
                self.frame_ready.emit(frame)
            
            self.msleep(16)  # ~60 FPS UI update
            
    def stop(self):
        """Stop the rendering thread"""
        self.running = False
        self.raytracer.stop_rendering()
        self.wait()


class ImageDisplay(QLabel):
    """Custom image display widget with mouse interaction"""
    mouse_moved = pyqtSignal(float, float)
    mouse_pressed = pyqtSignal(float, float)
    mouse_released = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #444; background-color: #1a1a1a;")
        self.setMinimumSize(400, 300)
        
        self.dragging = False
        self.last_pos = None
        
    def set_image(self, image_array):
        """Set image from numpy array"""
        if image_array is None or image_array.size == 0:
            return
            
        # Convert numpy array to QImage
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        
        # Convert to 8-bit
        image_8bit = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image))
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_pos = event.pos()
            # Convert to normalized coordinates
            if self.pixmap():
                pixmap_size = self.pixmap().size()
                label_size = self.size()
                
                # Calculate offset to center the image
                x_offset = (label_size.width() - pixmap_size.width()) / 2
                y_offset = (label_size.height() - pixmap_size.height()) / 2
                
                # Normalize coordinates
                norm_x = (event.x() - x_offset) / pixmap_size.width()
                norm_y = (event.y() - y_offset) / pixmap_size.height()
                
                if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                    self.mouse_pressed.emit(norm_x, norm_y)
                    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.last_pos = None
            self.mouse_released.emit()
            
    def mouseMoveEvent(self, event):
        if self.dragging and self.last_pos and self.pixmap():
            current_pos = event.pos()
            dx = current_pos.x() - self.last_pos.x()
            dy = current_pos.y() - self.last_pos.y()
            
            # Normalize movement
            if self.pixmap():
                pixmap_size = self.pixmap().size()
                norm_dx = dx / pixmap_size.width()
                norm_dy = dy / pixmap_size.height()
                
                self.mouse_moved.emit(norm_dx, norm_dy)
                
            self.last_pos = current_pos


class ControlPanel(QWidget):
    """Control panel with all settings"""
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, raytracer):
        super().__init__()
        self.raytracer = raytracer
        self.setup_ui()
        # Initialize object info after UI is setup
        self.update_object_info()
        
    def setup_ui(self):
        """Setup the control panel UI - UPDATED VERSION"""
        layout = QVBoxLayout()
        
        # Rendering settings
        render_group = self.create_render_group()
        layout.addWidget(render_group)
        
        # Camera controls - ADD THIS
        camera_group = self.create_camera_group()
        layout.addWidget(camera_group)
        
        # Object controls
        object_group = self.create_object_group()
        layout.addWidget(object_group)
        
        # Material controls
        material_group = self.create_material_group()
        layout.addWidget(material_group)
        
        # Denoiser controls
        denoiser_group = self.create_denoiser_group()
        layout.addWidget(denoiser_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def create_render_group(self):
        """Create rendering controls group"""
        group = QGroupBox("Rendering Settings")
        layout = QVBoxLayout()
        
        # Max Samples
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Max Samples:"))
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 1024)
        self.max_samples.setValue(self.raytracer.settings["max_samples"])
        self.max_samples.valueChanged.connect(self.on_settings_changed)
        samples_layout.addWidget(self.max_samples)
        layout.addLayout(samples_layout)
        
        # Samples per Batch
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Samples/Batch:"))
        self.samples_batch = QSpinBox()
        self.samples_batch.setRange(1, 64)
        self.samples_batch.setValue(self.raytracer.settings["samples_per_batch"])
        self.samples_batch.valueChanged.connect(self.on_settings_changed)
        batch_layout.addWidget(self.samples_batch)
        layout.addLayout(batch_layout)
        
        # Max Depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 32)
        self.max_depth.setValue(self.raytracer.settings["max_depth"])
        self.max_depth.valueChanged.connect(self.on_settings_changed)
        depth_layout.addWidget(self.max_depth)
        layout.addLayout(depth_layout)
        
        # Exposure
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Exposure:"))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.1, 5.0)
        self.exposure.setSingleStep(0.1)
        self.exposure.setValue(self.raytracer.settings["exposure"])
        self.exposure.valueChanged.connect(self.on_settings_changed)
        exposure_layout.addWidget(self.exposure)
        layout.addLayout(exposure_layout)
        
        # Move Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Move Speed:"))
        self.move_speed = QDoubleSpinBox()
        self.move_speed.setRange(0.1, 2.0)
        self.move_speed.setSingleStep(0.1)
        self.move_speed.setValue(self.raytracer.settings["move_speed"])
        self.move_speed.valueChanged.connect(self.on_settings_changed)
        speed_layout.addWidget(self.move_speed)
        layout.addLayout(speed_layout)
        
        group.setLayout(layout)
        return group
    

    def create_camera_group(self):
        """Create camera controls group"""
        group = QGroupBox("Camera Controls")
        layout = QVBoxLayout()
        
        # Camera movement buttons
        move_layout = QVBoxLayout()
        move_label = QLabel("Camera Movement:")
        move_label.setStyleSheet("font-weight: bold;")
        move_layout.addWidget(move_label)
        
        # Camera movement grid
        cam_grid = QHBoxLayout()
        
        # Left/Right
        cam_lr_layout = QVBoxLayout()
        self.cam_btn_left = QPushButton("← Left")
        self.cam_btn_right = QPushButton("Right →")
        cam_lr_layout.addWidget(self.cam_btn_left)
        cam_lr_layout.addWidget(self.cam_btn_right)
        
        # Up/Down  
        cam_ud_layout = QVBoxLayout()
        self.cam_btn_up = QPushButton("↑ Up")
        self.cam_btn_down = QPushButton("Down ↓")
        cam_ud_layout.addWidget(self.cam_btn_up)
        cam_ud_layout.addWidget(self.cam_btn_down)
        
        # Forward/Backward
        cam_fb_layout = QVBoxLayout()
        self.cam_btn_forward = QPushButton("↗ Forward")
        self.cam_btn_backward = QPushButton("Backward ↙")
        cam_fb_layout.addWidget(self.cam_btn_forward)
        cam_fb_layout.addWidget(self.cam_btn_backward)
        
        cam_grid.addLayout(cam_lr_layout)
        cam_grid.addLayout(cam_ud_layout) 
        cam_grid.addLayout(cam_fb_layout)
        
        move_layout.addLayout(cam_grid)
        
        # Camera info
        self.camera_info = QLabel("Camera: (0.00, 2.00, 3.00)")
        self.camera_info.setStyleSheet("color: #aaa; font-size: 10px;")
        move_layout.addWidget(self.camera_info)
        
        layout.addLayout(move_layout)
        
        # Connect camera buttons
        self.cam_btn_left.clicked.connect(lambda: self.debounced_camera_move(-1, 0, 0))
        self.cam_btn_right.clicked.connect(lambda: self.debounced_camera_move(1, 0, 0))
        self.cam_btn_up.clicked.connect(lambda: self.debounced_camera_move(0, 1, 0))
        self.cam_btn_down.clicked.connect(lambda: self.debounced_camera_move(0, -1, 0))
        self.cam_btn_forward.clicked.connect(lambda: self.debounced_camera_move(0, 0, 1))
        self.cam_btn_backward.clicked.connect(lambda: self.debounced_camera_move(0, 0, -1))
        
        group.setLayout(layout)
        return group
    
    def debounced_camera_move(self, dx, dy, dz):
        """Debounced camera movement - FIXED VERSION"""
        try:
            self.raytracer.move_camera(dx, dy, dz)
            self.update_camera_info()
            self.disable_camera_buttons()
            QTimer.singleShot(300, self.enable_camera_buttons)  # Longer delay
        except Exception as e:
            print(f"Camera move error: {e}")
    
    def disable_camera_buttons(self):
        """Disable camera buttons briefly"""
        for btn in [self.cam_btn_left, self.cam_btn_right, self.cam_btn_up,
                   self.cam_btn_down, self.cam_btn_forward, self.cam_btn_backward]:
            btn.setEnabled(False)
    
    def enable_camera_buttons(self):
        """Re-enable camera buttons"""
        for btn in [self.cam_btn_left, self.cam_btn_right, self.cam_btn_up,
                   self.cam_btn_down, self.cam_btn_forward, self.cam_btn_backward]:
            btn.setEnabled(True)
    
    def update_camera_info(self):
        """Update camera position info - FIXED VERSION"""
        try:
            camera = self.raytracer.camera
            if camera:
                pos = camera.position
                info = f"Camera: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})"
                self.camera_info.setText(info)
        except:
            self.camera_info.setText("Camera: (0.0, 2.0, 5.0)")  # Fallback

        
    def create_object_group(self):
        """Create object controls group - FIXED VERSION"""
        group = QGroupBox("Object Controls")
        layout = QVBoxLayout()
        
        # Object selection
        obj_layout = QHBoxLayout()
        obj_layout.addWidget(QLabel("Object:"))
        self.object_select = QComboBox()
        
        # Add objects including ground
        object_count = self.raytracer.get_object_count() + 1  # +1 for ground
        for i in range(object_count):
            obj = self.raytracer.scene.spheres[i] if i < len(self.raytracer.scene.spheres) else None
            if obj and hasattr(obj, 'name') and obj.name:
                self.object_select.addItem(obj.name)
            else:
                name = "Ground" if i == 0 else f"Object {i}"
                self.object_select.addItem(name)
        
        self.object_select.currentIndexChanged.connect(self.on_object_selected)
        obj_layout.addWidget(self.object_select)
        layout.addLayout(obj_layout)
        
        # Object info
        self.object_info = QLabel("Selected: Red Metallic")
        self.object_info.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(self.object_info)
        
        # Movement buttons - FIXED VERSION
        move_layout = QVBoxLayout()
        move_label = QLabel("Movement:")
        move_label.setStyleSheet("font-weight: bold;")
        move_layout.addWidget(move_label)
        
        # Movement grid
        move_grid = QHBoxLayout()
        
        # Left/Right
        lr_layout = QVBoxLayout()
        self.btn_left = QPushButton("← Left")
        self.btn_right = QPushButton("Right →")
        lr_layout.addWidget(self.btn_left)
        lr_layout.addWidget(self.btn_right)
        
        # Up/Down  
        ud_layout = QVBoxLayout()
        self.btn_up = QPushButton("↑ Up")
        self.btn_down = QPushButton("Down ↓")
        ud_layout.addWidget(self.btn_up)
        ud_layout.addWidget(self.btn_down)
        
        # Forward/Backward
        fb_layout = QVBoxLayout()
        self.btn_forward = QPushButton("↗ Forward")
        self.btn_backward = QPushButton("Backward ↙")
        fb_layout.addWidget(self.btn_forward)
        fb_layout.addWidget(self.btn_backward)
        
        move_grid.addLayout(lr_layout)
        move_grid.addLayout(ud_layout) 
        move_grid.addLayout(fb_layout)
        
        move_layout.addLayout(move_grid)
        layout.addLayout(move_layout)
        
        # Connect buttons with debouncing
        self.btn_left.clicked.connect(lambda: self.debounced_move(-1, 0, 0))
        self.btn_right.clicked.connect(lambda: self.debounced_move(1, 0, 0))
        self.btn_up.clicked.connect(lambda: self.debounced_move(0, 1, 0))
        self.btn_down.clicked.connect(lambda: self.debounced_move(0, -1, 0))
        self.btn_forward.clicked.connect(lambda: self.debounced_move(0, 0, -1))
        self.btn_backward.clicked.connect(lambda: self.debounced_move(0, 0, 1))
        
        group.setLayout(layout)
        return group
    
    def debounced_move(self, dx, dy, dz):
        """Debounced movement to prevent rapid successive calls"""
        self.move_object(dx, dy, dz)
        # Disable buttons briefly to prevent spam
        for btn in [self.btn_left, self.btn_right, self.btn_up, 
                   self.btn_down, self.btn_forward, self.btn_backward]:
            btn.setEnabled(False)
        QTimer.singleShot(200, self.enable_move_buttons)  # Re-enable after 200ms
    
    def enable_move_buttons(self):
        """Re-enable movement buttons"""
        for btn in [self.btn_left, self.btn_right, self.btn_up,
                   self.btn_down, self.btn_forward, self.btn_backward]:
            btn.setEnabled(True)

        
    def create_material_group(self):
        """Create material controls group"""
        group = QGroupBox("Material Properties")
        layout = QVBoxLayout()
        
        # Metallic
        metallic_layout = QHBoxLayout()
        metallic_layout.addWidget(QLabel("Metallic:"))
        self.metallic = QSlider(Qt.Horizontal)
        self.metallic.setRange(0, 100)
        self.metallic.setValue(90)
        self.metallic.valueChanged.connect(lambda: self.on_material_changed('metallic'))
        metallic_layout.addWidget(self.metallic)
        layout.addLayout(metallic_layout)
        
        # Roughness
        roughness_layout = QHBoxLayout()
        roughness_layout.addWidget(QLabel("Roughness:"))
        self.roughness = QSlider(Qt.Horizontal)
        self.roughness.setRange(0, 100)
        self.roughness.setValue(10)
        self.roughness.valueChanged.connect(lambda: self.on_material_changed('roughness'))
        roughness_layout.addWidget(self.roughness)
        layout.addLayout(roughness_layout)
        
        # Color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color R:"))
        self.color_r = QSlider(Qt.Horizontal)
        self.color_r.setRange(0, 100)
        self.color_r.setValue(90)
        self.color_r.valueChanged.connect(lambda: self.on_material_changed('albedo'))
        color_layout.addWidget(self.color_r)
        layout.addLayout(color_layout)
        
        # Light intensity - FIXED VERSION
        light_layout = QHBoxLayout()
        light_layout.addWidget(QLabel("Light Power:"))
        self.light_intensity = QDoubleSpinBox()  # Changed to QDoubleSpinBox
        self.light_intensity.setRange(0.1, 100.0)  # Allow decimal values
        self.light_intensity.setSingleStep(0.5)  # Smaller steps
        self.light_intensity.setValue(15.0)
        self.light_intensity.setDecimals(1)  # Allow decimal input
        self.light_intensity.setKeyboardTracking(False)  # Don't update on every keystroke
        self.light_intensity.valueChanged.connect(self.on_light_intensity_changed)
        light_layout.addWidget(self.light_intensity)
        layout.addLayout(light_layout)
        
        group.setLayout(layout)
        return group
        
    def create_denoiser_group(self):
        """Create denoiser controls group"""
        group = QGroupBox("Denoiser Settings")
        layout = QVBoxLayout()
        
        # Show denoisers
        self.show_denoisers = QCheckBox("Show Denoisers")
        self.show_denoisers.toggled.connect(self.on_show_denoisers_changed)
        layout.addWidget(self.show_denoisers)
        
        # Denoiser selection
        denoiser_layout = QVBoxLayout()
        denoiser_layout.addWidget(QLabel("Denoiser Methods:"))
        
        self.denoiser_bilateral = QCheckBox("Bilateral")
        self.denoiser_bilateral.setChecked(True)
        self.denoiser_bilateral.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_layout.addWidget(self.denoiser_bilateral)
        
        self.denoiser_nlmeans = QCheckBox("NL-Means")
        self.denoiser_nlmeans.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_layout.addWidget(self.denoiser_nlmeans)
        
        self.denoiser_gaussian = QCheckBox("Gaussian")
        self.denoiser_gaussian.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_layout.addWidget(self.denoiser_gaussian)
        
        self.denoiser_median = QCheckBox("Median")
        self.denoiser_median.toggled.connect(self.on_denoiser_selection_changed)
        denoiser_layout.addWidget(self.denoiser_median)
        
        layout.addLayout(denoiser_layout)
        group.setLayout(layout)
        return group
        
    def on_settings_changed(self):
        """Handle settings changes"""
        settings = {
            'max_samples': self.max_samples.value(),
            'samples_per_batch': self.samples_batch.value(),
            'max_depth': self.max_depth.value(),
            'exposure': self.exposure.value(),
            'move_speed': self.move_speed.value(),
        }
        self.settings_changed.emit(settings)
        
    def on_object_selected(self, index):
        """Handle object selection"""
        self.raytracer.settings['selected_object'] = index
        self.update_object_info()
        self.update_material_sliders()
        
    def update_object_info(self):
        """Update object information"""
        obj = self.raytracer.get_selected_object()
        if obj:
            name = getattr(obj, 'name', f'Object {self.object_select.currentIndex()}')
            self.object_info.setText(f"Selected: {name}")
            
    def update_material_sliders(self):
        """Update material sliders to match selected object - FIXED VERSION"""
        obj = self.raytracer.get_selected_object()
        if obj:
            mat = obj.material
            
            # Block signals temporarily to prevent recursive updates
            self.metallic.blockSignals(True)
            self.roughness.blockSignals(True)
            self.color_r.blockSignals(True)
            self.light_intensity.blockSignals(True)
            
            # Update sliders
            self.metallic.setValue(int(mat.metallic * 100))
            self.roughness.setValue(int(mat.roughness * 100))
            
            # For color, use average of RGB for the slider
            if hasattr(mat.albedo, 'x'):
                avg_color = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0
                self.color_r.setValue(int(avg_color * 100))
            
            # Update light intensity if it's a light
            if hasattr(mat, 'emission') and hasattr(mat.emission, 'x'):
                emission = mat.emission
                avg_emission = (emission.x + emission.y + emission.z) / 3.0
                self.light_intensity.setValue(avg_emission)
                self.light_intensity.setEnabled(avg_emission > 0.1)
            else:
                self.light_intensity.setEnabled(False)
                
            # Re-enable signals
            self.metallic.blockSignals(False)
            self.roughness.blockSignals(False)
            self.color_r.blockSignals(False)
            self.light_intensity.blockSignals(False)
                
    def move_object(self, dx, dy, dz):
        """Move selected object"""
        self.raytracer.move_object(dx, dy, dz)
        
    def on_material_changed(self, property_name):
        """Handle material property changes"""
        if property_name == 'albedo':
            value = self.color_r.value() / 100.0
        elif property_name in ['metallic', 'roughness']:
            value = getattr(self, property_name).value() / 100.0
        else:
            return
            
        self.raytracer.update_object_material(property_name, value)
        
    def on_light_intensity_changed(self, value):
        """Handle light intensity changes"""
        self.raytracer.update_light_intensity(value)
        
    def on_show_denoisers_changed(self, checked):
        """Handle show denoisers toggle"""
        self.raytracer.settings['show_denoisers'] = checked
        
    def on_denoiser_selection_changed(self):
        """Handle denoiser selection changes"""
        selected = []
        if self.denoiser_bilateral.isChecked():
            selected.append('bilateral')
        if self.denoiser_nlmeans.isChecked():
            selected.append('nlmeans')
        if self.denoiser_gaussian.isChecked():
            selected.append('gaussian')
        if self.denoiser_median.isChecked():
            selected.append('median')
            
        self.raytracer.settings['selected_denoisers'] = selected


class GUI(QMainWindow):
    """Main application window - maintains original interface"""
    
    def __init__(self):
        super().__init__()
        self.raytracer = RayTracerInteraction(640, 480)
        self.render_thread = None
        self.setup_ui()
        self.setup_rendering()
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("C++ Ray Tracer - Interactive Controls")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side - Image displays
        left_widget = self.create_image_displays()
        main_layout.addWidget(left_widget, 3)
        
        # Right side - Controls
        right_widget = self.create_control_panel()
        main_layout.addWidget(right_widget, 1)
        
        # Status bar
        self.status_label = QLabel("Ready to render...")
        self.statusBar().addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #88c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #88c;
                border: 1px solid #55a;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #666;
                background: #333;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #88c;
                background: #55a;
            }
            QComboBox {
                border: 1px solid #555;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                min-width: 6em;
                background: #333;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #555;
                border-radius: 3px;
                padding: 1px;
                background: #333;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #333;
                color: #aaa;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #444;
                color: #fff;
                border-bottom: 2px solid #88c;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
    def create_image_displays(self):
        """Create the image display area"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Main view tab
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        main_tab.setLayout(main_layout)
        
        self.main_display = ImageDisplay()
        self.main_display.mouse_moved.connect(self.on_mouse_drag)
        self.main_display.mouse_pressed.connect(self.on_mouse_press)
        self.main_display.mouse_released.connect(self.on_mouse_release)
        main_layout.addWidget(self.main_display)
        
        self.tabs.addTab(main_tab, "Main View")
        
        # Enhanced view tab
        enhanced_tab = QWidget()
        enhanced_layout = QVBoxLayout()
        enhanced_tab.setLayout(enhanced_layout)
        
        self.enhanced_display = ImageDisplay()
        enhanced_layout.addWidget(self.enhanced_display)
        
        self.tabs.addTab(enhanced_tab, "Enhanced View")
        
        # Denoiser views tab
        denoiser_tab = QWidget()
        denoiser_layout = QVBoxLayout()
        denoiser_tab.setLayout(denoiser_layout)
        
        # Grid layout for denoiser displays
        denoiser_grid = QHBoxLayout()
        
        self.denoiser_displays = {}
        methods = ['bilateral', 'nlmeans', 'gaussian', 'median']
        for method in methods:
            display_widget = QWidget()
            display_layout = QVBoxLayout()
            display_widget.setLayout(display_layout)
            
            label = QLabel(f"{method.title()} Denoised")
            label.setAlignment(Qt.AlignCenter)
            display_layout.addWidget(label)
            
            display = ImageDisplay()
            display.setMinimumSize(300, 200)
            display_layout.addWidget(display)
            
            denoiser_grid.addWidget(display_widget)
            self.denoiser_displays[method] = display
            
        denoiser_layout.addLayout(denoiser_grid)
        self.tabs.addTab(denoiser_tab, "Denoiser Views")
        
        layout.addWidget(self.tabs)
        return widget
        
    def create_control_panel(self):
        """Create the control panel"""
        self.control_panel = ControlPanel(self.raytracer)
        self.control_panel.settings_changed.connect(self.on_settings_changed)
        return self.control_panel
        
    def setup_rendering(self):
        """Setup rendering thread"""
        self.render_thread = RenderThread(self.raytracer)
        self.render_thread.frame_ready.connect(self.on_frame_ready)
        self.render_thread.rendering_finished.connect(self.on_rendering_finished)
        self.render_thread.start()
        
    def on_frame_ready(self, frame_data):
        """Handle new frame from render thread"""
        # Update main display
        self.main_display.set_image(frame_data['display'])
        self.enhanced_display.set_image(frame_data['enhanced'])
        
        # Update denoiser displays
        if 'denoised' in frame_data:
            for method, image in frame_data['denoised'].items():
                if method in self.denoiser_displays:
                    self.denoiser_displays[method].set_image(image)
        
        # Update status
        info = (f"Samples: {frame_data['samples']} | "
               f"Batch Time: {frame_data['render_time']:.3f}s | "
               f"FPS: {1/frame_data['render_time']:.1f}" if frame_data['render_time'] > 0 else "Rendering...")
        self.status_label.setText(info)
        
        # Update progress
        max_samples = self.raytracer.settings['max_samples']
        progress = min(100, int((frame_data['samples'] / max_samples) * 100))
        self.progress_bar.setValue(progress)
        self.progress_bar.setVisible(progress < 100)
        
    def on_rendering_finished(self):
        """Handle rendering completion"""
        self.status_label.setText("Rendering Complete!")
        self.progress_bar.setVisible(False)
        
    def on_settings_changed(self, settings):
        """Handle settings changes"""
        self.raytracer.settings.update(settings)
        self.raytracer.restart_rendering()
        
    def on_mouse_press(self, x, y):
        """Handle mouse press - UPDATED VERSION"""
        # Try object selection first
        if self.raytracer.select_object_by_click(x, y):
            # Update control panel to reflect new selection
            self.control_panel.object_select.setCurrentIndex(self.raytracer.settings['selected_object'])
            self.control_panel.update_object_info()
            self.control_panel.update_material_sliders()
        else:
            # If no object selected, start camera dragging
            self.raytracer.camera_dragging = True
    
    def on_mouse_drag(self, dx, dy):
        """Handle mouse dragging - UPDATED VERSION"""
        if self.raytracer.dragging:
            # Object dragging
            speed = 2.0
            self.raytracer.move_object(dx * speed, -dy * speed, 0)
        elif self.raytracer.camera_dragging:
            # Camera rotation
            self.raytracer.rotate_camera(dx, -dy)
            
    def on_mouse_release(self):
        """Handle mouse release - UPDATED VERSION"""
        self.raytracer.dragging = False
        self.raytracer.camera_dragging = False
        
    def keyPressEvent(self, event):
        """Handle keyboard input"""
        key = event.key()
        move_dict = {
            Qt.Key_Left: (-1, 0, 0), Qt.Key_Right: (1, 0, 0),
            Qt.Key_Up: (0, 1, 0), Qt.Key_Down: (0, -1, 0),
            Qt.Key_W: (0, 0, -1), Qt.Key_S: (0, 0, 1),
            Qt.Key_A: (-1, 0, 0), Qt.Key_D: (1, 0, 0),
            Qt.Key_Q: (0, 1, 0), Qt.Key_E: (0, -1, 0)
        }
        
        if key in move_dict:
            self.raytracer.move_object(*move_dict[key])
            
    def closeEvent(self, event):
        """Handle application close"""
        if self.render_thread:
            self.render_thread.stop()
        event.accept()
    
