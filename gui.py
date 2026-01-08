#gui.py

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QSlider, QCheckBox, QComboBox, QLabel, QPushButton,
                             QTabWidget, QSplitter, QProgressBar, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QKeyEvent
import cv2

from interaction import RayTracerInteraction, RenderMode

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
            while self.raytracer.has_frames():
                frame = self.raytracer.get_frame()
                if frame is None:
                    break
                
                if 'done' in frame:
                    self.rendering_finished.emit()
                    break
                
                self.frame_ready.emit(frame)
            
            self.msleep(16)  # ~60 FPS
    
    def stop(self):
        """Stop the rendering thread"""
        self.running = False
        self.raytracer.stop_rendering()
        self.wait()

class ImageDisplay(QLabel):
    """Custom image display widget with mouse interaction"""
    mouse_moved = pyqtSignal(float, float)
    mouse_pressed = pyqtSignal(float, float, int)
    mouse_released = pyqtSignal(int)
    right_click = pyqtSignal(float, float)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #444; background-color: #1a1a1a;")
        self.setMinimumSize(400, 300)
        
        self.dragging = False
        self.drag_button = None
        self.last_pos = None
    
    def set_image(self, image_array):
        """Set image from numpy array"""
        if image_array is None or image_array.size == 0:
            return
        
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        
        image_8bit = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_image))
    
    def mousePressEvent(self, event):
        button = event.button()
        if button in [Qt.LeftButton, Qt.RightButton]:
            self.dragging = True
            self.drag_button = button
            self.last_pos = event.pos()
            
            if self.pixmap():
                pixmap_size = self.pixmap().size()
                label_size = self.size()
                
                x_offset = (label_size.width() - pixmap_size.width()) / 2
                y_offset = (label_size.height() - pixmap_size.height()) / 2
                
                norm_x = (event.x() - x_offset) / pixmap_size.width()
                norm_y = (event.y() - y_offset) / pixmap_size.height()
                
                if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                    if button == Qt.RightButton:
                        self.right_click.emit(norm_x, norm_y)
                    self.mouse_pressed.emit(norm_x, norm_y, button)
    
    def mouseReleaseEvent(self, event):
        button = event.button()
        if button == self.drag_button:
            self.dragging = False
            self.drag_button = None
            self.last_pos = None
            self.mouse_released.emit(button)
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.last_pos and self.pixmap():
            current_pos = event.pos()
            dx = current_pos.x() - self.last_pos.x()
            dy = current_pos.y() - self.last_pos.y()
            
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
        self.update_object_info()
        self.update_camera_info()
    
    def setup_ui(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout()
        
        # Rendering settings
        render_group = self.create_render_group()
        layout.addWidget(render_group)
        
        # Camera settings
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
        """Create rendering controls"""
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
        
        group.setLayout(layout)
        return group
    
    def create_camera_group(self):
        """Create camera controls"""
        group = QGroupBox("Camera Settings")
        layout = QVBoxLayout()
        
        # Camera position
        pos_group = QGroupBox("Position")
        pos_layout = QVBoxLayout()
        
        # X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.cam_x = QDoubleSpinBox()
        self.cam_x.setRange(-20, 20)
        self.cam_x.setSingleStep(0.1)
        self.cam_x.setValue(0.0)
        self.cam_x.valueChanged.connect(self.on_camera_pos_changed)
        x_layout.addWidget(self.cam_x)
        pos_layout.addLayout(x_layout)
        
        # Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.cam_y = QDoubleSpinBox()
        self.cam_y.setRange(-20, 20)
        self.cam_y.setSingleStep(0.1)
        self.cam_y.setValue(2.0)
        self.cam_y.valueChanged.connect(self.on_camera_pos_changed)
        y_layout.addWidget(self.cam_y)
        pos_layout.addLayout(y_layout)
        
        # Z
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.cam_z = QDoubleSpinBox()
        self.cam_z.setRange(-20, 20)
        self.cam_z.setSingleStep(0.1)
        self.cam_z.setValue(5.0)
        self.cam_z.valueChanged.connect(self.on_camera_pos_changed)
        z_layout.addWidget(self.cam_z)
        pos_layout.addLayout(z_layout)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Camera target
        target_group = QGroupBox("Target")
        target_layout = QVBoxLayout()
        
        # Target X
        tx_layout = QHBoxLayout()
        tx_layout.addWidget(QLabel("X:"))
        self.target_x = QDoubleSpinBox()
        self.target_x.setRange(-20, 20)
        self.target_x.setSingleStep(0.1)
        self.target_x.setValue(0.0)
        self.target_x.valueChanged.connect(self.on_camera_target_changed)
        tx_layout.addWidget(self.target_x)
        target_layout.addLayout(tx_layout)
        
        # Target Y
        ty_layout = QHBoxLayout()
        ty_layout.addWidget(QLabel("Y:"))
        self.target_y = QDoubleSpinBox()
        self.target_y.setRange(-20, 20)
        self.target_y.setSingleStep(0.1)
        self.target_y.setValue(0.0)
        self.target_y.valueChanged.connect(self.on_camera_target_changed)
        ty_layout.addWidget(self.target_y)
        target_layout.addLayout(ty_layout)
        
        # Target Z
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Z:"))
        self.target_z = QDoubleSpinBox()
        self.target_z.setRange(-20, 20)
        self.target_z.setSingleStep(0.1)
        self.target_z.setValue(-1.0)
        self.target_z.valueChanged.connect(self.on_camera_target_changed)
        tz_layout.addWidget(self.target_z)
        target_layout.addLayout(tz_layout)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # FOV
        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("FOV:"))
        self.fov = QDoubleSpinBox()
        self.fov.setRange(10, 120)
        self.fov.setSingleStep(1.0)
        self.fov.setValue(45.0)
        self.fov.valueChanged.connect(self.on_camera_fov_changed)
        fov_layout.addWidget(self.fov)
        layout.addLayout(fov_layout)
        
        # Speed controls
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Move Speed:"))
        self.move_speed = QDoubleSpinBox()
        self.move_speed.setRange(0.01, 1.0)
        self.move_speed.setSingleStep(0.01)
        self.move_speed.setValue(self.raytracer.settings["camera_move_speed"])
        self.move_speed.valueChanged.connect(self.on_move_speed_changed)
        speed_layout.addWidget(self.move_speed)
        layout.addLayout(speed_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self.reset_camera)
        layout.addWidget(reset_btn)
        
        group.setLayout(layout)
        return group
    
    def create_object_group(self):
        """Create object controls"""
        group = QGroupBox("Object Controls")
        layout = QVBoxLayout()
        
        # Object selection
        obj_layout = QHBoxLayout()
        obj_layout.addWidget(QLabel("Object:"))
        self.object_select = QComboBox()
        
        # Populate objects
        for i in range(self.raytracer.get_object_count() + 1):
            obj = self.raytracer.scene.spheres[i] if i < len(self.raytracer.scene.spheres) else None
            if obj and obj.name:
                self.object_select.addItem(obj.name)
            else:
                name = "Ground" if i == 0 else f"Object {i}"
                self.object_select.addItem(name)
        
        self.object_select.currentIndexChanged.connect(self.on_object_selected)
        obj_layout.addWidget(self.object_select)
        layout.addLayout(obj_layout)
        
        # Object info
        self.object_info = QLabel("Selected: None")
        self.object_info.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(self.object_info)
        
        # Movement buttons
        move_group = QGroupBox("Keyboard Movement")
        move_layout = QVBoxLayout()
        
        # Horizontal movement
        horiz_layout = QHBoxLayout()
        self.btn_left = QPushButton("← Left (A)")
        self.btn_right = QPushButton("Right (D) →")
        horiz_layout.addWidget(self.btn_left)
        horiz_layout.addWidget(self.btn_right)
        move_layout.addLayout(horiz_layout)
        
        # Vertical movement
        vert_layout = QHBoxLayout()
        self.btn_up = QPushButton("↑ Up (W)")
        self.btn_down = QPushButton("Down (S) ↓")
        vert_layout.addWidget(self.btn_up)
        vert_layout.addWidget(self.btn_down)
        move_layout.addLayout(vert_layout)
        
        # Depth movement
        depth_layout = QHBoxLayout()
        self.btn_forward = QPushButton("↗ Forward (Q)")
        self.btn_backward = QPushButton("Backward (E) ↙")
        depth_layout.addWidget(self.btn_forward)
        depth_layout.addWidget(self.btn_backward)
        move_layout.addLayout(depth_layout)
        
        move_group.setLayout(move_layout)
        layout.addWidget(move_group)
        
        # Connect buttons
        self.btn_left.clicked.connect(lambda: self.raytracer.move_object(-1, 0, 0))
        self.btn_right.clicked.connect(lambda: self.raytracer.move_object(1, 0, 0))
        self.btn_up.clicked.connect(lambda: self.raytracer.move_object(0, 1, 0))
        self.btn_down.clicked.connect(lambda: self.raytracer.move_object(0, -1, 0))
        self.btn_forward.clicked.connect(lambda: self.raytracer.move_object(0, 0, -1))
        self.btn_backward.clicked.connect(lambda: self.raytracer.move_object(0, 0, 1))
        
        group.setLayout(layout)
        return group
    
    def create_material_group(self):
        """Create material controls"""
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
        color_layout.addWidget(QLabel("Color:"))
        self.color_r = QSlider(Qt.Horizontal)
        self.color_r.setRange(0, 100)
        self.color_r.setValue(90)
        self.color_r.valueChanged.connect(lambda: self.on_material_changed('albedo'))
        color_layout.addWidget(self.color_r)
        layout.addLayout(color_layout)
        
        # Light intensity
        light_layout = QHBoxLayout()
        light_layout.addWidget(QLabel("Light Power:"))
        self.light_intensity = QDoubleSpinBox()
        self.light_intensity.setRange(0.1, 100.0)
        self.light_intensity.setSingleStep(0.5)
        self.light_intensity.setValue(15.0)
        self.light_intensity.setDecimals(1)
        self.light_intensity.setKeyboardTracking(False)
        self.light_intensity.valueChanged.connect(self.on_light_intensity_changed)
        light_layout.addWidget(self.light_intensity)
        layout.addLayout(light_layout)
        
        group.setLayout(layout)
        return group
    
    def create_denoiser_group(self):
        """Create denoiser controls"""
        group = QGroupBox("Denoiser Settings")
        layout = QVBoxLayout()
        
        self.show_denoisers = QCheckBox("Show Denoisers")
        self.show_denoisers.toggled.connect(self.on_show_denoisers_changed)
        layout.addWidget(self.show_denoisers)
        
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
        }
        self.settings_changed.emit(settings)
    
    def on_camera_pos_changed(self):
        """Update camera position"""
        pos = Vector3(self.cam_x.value(), self.cam_y.value(), self.cam_z.value())
        self.raytracer.camera.position = pos
        self.raytracer.update_camera_frame()
        self.raytracer.restart_rendering()
    
    def on_camera_target_changed(self):
        """Update camera target"""
        target = Vector3(self.target_x.value(), self.target_y.value(), self.target_z.value())
        self.raytracer.camera.target = target
        self.raytracer.update_camera_frame()
        self.raytracer.restart_rendering()
    
    def on_camera_fov_changed(self):
        """Update camera FOV"""
        self.raytracer.camera.fov = self.fov.value()
        self.raytracer.restart_rendering()
    
    def on_move_speed_changed(self):
        """Update camera movement speed"""
        self.raytracer.settings['camera_move_speed'] = self.move_speed.value()
    
    def reset_camera(self):
        """Reset camera to default"""
        self.cam_x.setValue(0.0)
        self.cam_y.setValue(2.0)
        self.cam_z.setValue(5.0)
        self.target_x.setValue(0.0)
        self.target_y.setValue(0.0)
        self.target_z.setValue(-1.0)
        self.fov.setValue(45.0)
    
    def update_camera_info(self):
        """Update camera controls from current camera"""
        camera = self.raytracer.camera
        if camera:
            self.cam_x.blockSignals(True)
            self.cam_y.blockSignals(True)
            self.cam_z.blockSignals(True)
            self.target_x.blockSignals(True)
            self.target_y.blockSignals(True)
            self.target_z.blockSignals(True)
            self.fov.blockSignals(True)
            
            self.cam_x.setValue(camera.position.x)
            self.cam_y.setValue(camera.position.y)
            self.cam_z.setValue(camera.position.z)
            self.target_x.setValue(camera.target.x)
            self.target_y.setValue(camera.target.y)
            self.target_z.setValue(camera.target.z)
            self.fov.setValue(camera.fov)
            
            self.cam_x.blockSignals(False)
            self.cam_y.blockSignals(False)
            self.cam_z.blockSignals(False)
            self.target_x.blockSignals(False)
            self.target_y.blockSignals(False)
            self.target_z.blockSignals(False)
            self.fov.blockSignals(False)
    
    def on_object_selected(self, index):
        """Handle object selection"""
        self.raytracer.settings['selected_object'] = index
        self.update_object_info()
        self.update_material_sliders()
    
    def update_object_info(self):
        """Update object information display"""
        obj = self.raytracer.get_selected_object()
        if obj:
            self.object_info.setText(f"Selected: {obj.name}")
    
    def update_material_sliders(self):
        """Update material sliders to match selected object"""
        obj = self.raytracer.get_selected_object()
        if obj:
            mat = obj.material
            
            self.metallic.blockSignals(True)
            self.roughness.blockSignals(True)
            self.color_r.blockSignals(True)
            self.light_intensity.blockSignals(True)
            
            self.metallic.setValue(int(mat.metallic * 100))
            self.roughness.setValue(int(mat.roughness * 100))
            
            if hasattr(mat.albedo, 'x'):
                avg_color = (mat.albedo.x + mat.albedo.y + mat.albedo.z) / 3.0
                self.color_r.setValue(int(avg_color * 100))
            
            if hasattr(mat, 'emission') and hasattr(mat.emission, 'x'):
                emission = mat.emission
                avg_emission = (emission.x + emission.y + emission.z) / 3.0
                self.light_intensity.setValue(avg_emission)
                self.light_intensity.setEnabled(avg_emission > 0.1)
            else:
                self.light_intensity.setEnabled(False)
            
            self.metallic.blockSignals(False)
            self.roughness.blockSignals(False)
            self.color_r.blockSignals(False)
            self.light_intensity.blockSignals(False)
    
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
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.raytracer = RayTracerInteraction(640, 480)
        self.render_thread = None
        
        # Camera key mapping
        self.camera_keys = {
            Qt.Key_W: 'forward',
            Qt.Key_S: 'backward',
            Qt.Key_A: 'left',
            Qt.Key_D: 'right',
            Qt.Key_Space: 'up',
            Qt.Key_Control: 'down',
            Qt.Key_Shift: 'up',  # Alternative up
        }
        
        # Object dragging state
        self.dragging_object = False
        self.dimension_locks = {'x': False, 'y': False, 'z': False}
        
        # Track manual mode changes vs automatic ones
        self.manual_mode_change = False
        
        self.setup_ui()
        self.setup_rendering()

        self.camera_update_timer = QTimer()
        self.camera_update_timer.timeout.connect(self.update_camera_controls)
        self.camera_update_timer.start(100)  # Update every 100ms
    
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
        
        # Mode indicator
        self.mode_label = QLabel("Mode: Ray Tracing")
        self.mode_label.setStyleSheet("color: #88c; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.mode_label)
        
        # Lock status
        self.lock_label = QLabel("Locks: None")
        self.statusBar().addPermanentWidget(self.lock_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Focus policy
        self.setFocusPolicy(Qt.StrongFocus)
    
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
            QPushButton {
                background-color: #444;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
                color: white;
            }
            QPushButton:hover {
                background-color: #555;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)
    
    def create_image_displays(self):
        """Create the image display area"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Mode selector
        mode_widget = QWidget()
        mode_layout = QHBoxLayout()
        mode_widget.setLayout(mode_layout)
        
        self.raytrace_btn = QPushButton("Ray Tracing")
        self.raytrace_btn.setCheckable(True)
        self.raytrace_btn.setChecked(True)
        self.raytrace_btn.clicked.connect(self.on_raytrace_mode)
        mode_layout.addWidget(self.raytrace_btn)
        
        self.wireframe_btn = QPushButton("Wireframe")
        self.wireframe_btn.setCheckable(True)
        self.wireframe_btn.clicked.connect(self.on_wireframe_mode)
        mode_layout.addWidget(self.wireframe_btn)
        
        self.silhouette_btn = QPushButton("Silhouette")
        self.silhouette_btn.setCheckable(True)
        self.silhouette_btn.clicked.connect(self.on_silhouette_mode)
        mode_layout.addWidget(self.silhouette_btn)
        
        mode_layout.addStretch()
        layout.addWidget(mode_widget)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Main view tab
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        main_tab.setLayout(main_layout)
        
        self.main_display = ImageDisplay()
        self.main_display.mouse_pressed.connect(self.on_mouse_press)
        self.main_display.mouse_moved.connect(self.on_mouse_drag)
        self.main_display.mouse_released.connect(self.on_mouse_release)
        self.main_display.right_click.connect(self.on_right_click)
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
        
        # Instructions
        instructions = QLabel(
            "<b>Controls:</b> WASD+Space/Ctrl to move camera | "
            "<b>Right Click + Drag</b> to rotate camera | "
            "<b>Hold X/Y/Z + Left Click + Drag</b> to move object | "
            "<b>ESC</b> to cancel"
        )
        instructions.setStyleSheet("""
            QLabel {
                color: #aaa;
                font-size: 10px;
                padding: 5px;
                background-color: #222;
                border-radius: 3px;
            }
        """)
        layout.addWidget(instructions)
        
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

    def update_camera_controls(self):
        """Update camera control values from current camera state"""
        if self.raytracer.camera:
            camera = self.raytracer.camera
            
            # Update position controls
            self.control_panel.cam_x.blockSignals(True)
            self.control_panel.cam_y.blockSignals(True)
            self.control_panel.cam_z.blockSignals(True)
            
            self.control_panel.cam_x.setValue(camera.position.x)
            self.control_panel.cam_y.setValue(camera.position.y)
            self.control_panel.cam_z.setValue(camera.position.z)
            
            self.control_panel.cam_x.blockSignals(False)
            self.control_panel.cam_y.blockSignals(False)
            self.control_panel.cam_z.blockSignals(False)
            
            # Update target controls
            self.control_panel.target_x.blockSignals(True)
            self.control_panel.target_y.blockSignals(True)
            self.control_panel.target_z.blockSignals(True)
            
            self.control_panel.target_x.setValue(camera.target.x)
            self.control_panel.target_y.setValue(camera.target.y)
            self.control_panel.target_z.setValue(camera.target.z)
            
            self.control_panel.target_x.blockSignals(False)
            self.control_panel.target_y.blockSignals(False)
            self.control_panel.target_z.blockSignals(False)
            
            # Sync the ray tracer camera with the interactive camera
            rt_camera = self.raytracer.ray_tracer.get_camera()
            if rt_camera:
                rt_camera.position = camera.position
                rt_camera.target = camera.target
                rt_camera.up = camera.up
                rt_camera.fov = camera.fov
    
    def on_raytrace_mode(self):
        """Switch to ray tracing mode"""
        self.raytrace_btn.setChecked(True)
        self.wireframe_btn.setChecked(False)
        self.silhouette_btn.setChecked(False)
        self.current_mode = "raytracing"
        self.mode_label.setText("Mode: Ray Tracing")
        self.mode_label.setStyleSheet("color: #88c; font-weight: bold;")
        self.raytracer.render_mode = RenderMode.RAYTRACING
        self.raytracer.restart_rendering()
    
    def on_wireframe_mode(self):
        """Switch to wireframe mode - manual"""
        self.manual_mode_change = True
        self.raytrace_btn.setChecked(False)
        self.wireframe_btn.setChecked(True)
        self.silhouette_btn.setChecked(False)
        
        self.mode_label.setText("Mode: Wireframe")
        self.mode_label.setStyleSheet("color: #0f0; font-weight: bold;")
        self.raytracer.render_mode = RenderMode.WIREFRAME
        self.raytracer.previous_render_mode = RenderMode.WIREFRAME
        self.raytracer._process_frame_for_display(0.016)
        self.manual_mode_change = False
    
    def on_silhouette_mode(self):
        """Switch to silhouette mode - manual"""
        self.manual_mode_change = True
        self.raytrace_btn.setChecked(False)
        self.wireframe_btn.setChecked(False)
        self.silhouette_btn.setChecked(True)
        
        self.mode_label.setText("Mode: Silhouette")
        self.mode_label.setStyleSheet("color: #ff0; font-weight: bold;")
        self.raytracer.render_mode = RenderMode.SILHOUETTE
        self.raytracer.previous_render_mode = RenderMode.SILHOUETTE
        self.raytracer._process_frame_for_display(0.016)
        self.manual_mode_change = False
    
    def on_frame_ready(self, frame_data):
        """Handle new frame from render thread"""
        # Update displays
        self.main_display.set_image(frame_data['display'])
        self.enhanced_display.set_image(frame_data['enhanced'])
        
        # Update denoiser displays if needed
        if 'denoised' in frame_data:
            for method, image in frame_data['denoised'].items():
                if method in self.denoiser_displays:
                    self.denoiser_displays[method].set_image(image)
        
        # Update camera controls
        self.update_camera_controls()
        
        # Update status (rest of existing code remains the same)
        mode = frame_data.get('mode', 'raytracing')
        if mode == 'wireframe':
            status = "Wireframe Mode - Right Drag to Rotate, WASD to Move"
        elif mode == 'silhouette':
            if self.dragging_object:
                locks = self.get_lock_string()
                status = f"Dragging Object - Locks: {locks}"
            else:
                status = "Silhouette Mode - Hold X/Y/Z + Drag to Move Objects"
        else:
            if frame_data['is_raytracing']:
                status = (f"Samples: {frame_data['samples']} | "
                         f"Batch Time: {frame_data['render_time']:.3f}s")
            else:
                status = "Ray Tracing Mode"
        
        self.status_label.setText(status)
        
        # Update progress bar
        if frame_data.get('is_raytracing', False):
            max_samples = self.raytracer.settings['max_samples']
            progress = min(100, int((frame_data['samples'] / max_samples) * 100))
            self.progress_bar.setValue(progress)
            self.progress_bar.setVisible(progress < 100)
        else:
            self.progress_bar.setVisible(False)
    
    def on_rendering_finished(self):
        """Handle rendering completion"""
        self.status_label.setText("Rendering Complete!")
        self.progress_bar.setVisible(False)
    
    def on_settings_changed(self, settings):
        """Handle settings changes"""
        self.raytracer.settings.update(settings)
        self.raytracer.restart_rendering()
    
    def on_mouse_press(self, x: float, y: float, button: int):
        """Handle mouse press"""
        if button == Qt.LeftButton:
            # Check if any dimension is locked
            any_lock = any(self.dimension_locks.values())
            
            if any_lock:
                # Start object dragging
                if self.raytracer.start_object_dragging(x, y):
                    self.dragging_object = True
                    # Only switch to silhouette if not already in it
                    if not self.silhouette_btn.isChecked() and not self.manual_mode_change:
                        self.on_silhouette_mode()
            else:
                # Simple object selection
                if self.raytracer.select_object_by_click(x, y):
                    # Update control panel
                    idx = self.raytracer.settings['selected_object']
                    self.control_panel.object_select.setCurrentIndex(idx)
                    self.control_panel.update_object_info()
                    self.control_panel.update_material_sliders()
        
        elif button == Qt.RightButton:
            # Start camera rotation
            self.raytracer.start_camera_rotation(x, y)
            # Only switch to wireframe if not manually in another mode
            if not self.wireframe_btn.isChecked() and not self.manual_mode_change:
                self.on_wireframe_mode()
    
    def on_right_click(self, x: float, y: float):
        """Handle right click (alternative)"""
        self.raytracer.start_camera_rotation(x, y)
        if not self.wireframe_btn.isChecked():
            self.on_wireframe_mode()
    
    def on_mouse_drag(self, dx: float, dy: float):
        """Handle mouse dragging"""
        if self.dragging_object:
            # Object dragging
            self.raytracer.update_object_dragging(dx, dy)
            
            # Update object info
            obj = self.raytracer.get_selected_object()
            if obj:
                pos = obj.center
                self.control_panel.object_info.setText(
                    f"Dragging: {obj.name} at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})"
                )
        
        elif self.raytracer.camera_rotating:
            # Camera rotation
            self.raytracer.update_camera_rotation(dx, dy)
            self.control_panel.update_camera_info()
    
    def on_mouse_release(self, button: int):
        """Handle mouse release with proper mode restoration"""
        if button == Qt.LeftButton and self.dragging_object:
            self.raytracer.stop_object_dragging()
            self.dragging_object = False
            self.dimension_locks = {'x': False, 'y': False, 'z': False}
            self.update_lock_status()
            
            # Update control panel
            self.control_panel.update_object_info()
            self.control_panel.update_material_sliders()
            
            # Always return to ray tracing after dragging
            self.on_raytrace_mode()
        
        elif button == Qt.RightButton:
            self.raytracer.stop_camera_rotation()
            # Always return to ray tracing after camera rotation
            self.on_raytrace_mode()
    
    
    def keyPressEvent(self, event):
        """Handle keyboard input with better debouncing"""
        key = event.key()

        self.manual_mode_change = False
        
        # Camera movement keys
        if key in self.camera_keys:
            key_name = self.camera_keys[key]
            # Only send if key state is changing
            if not self.raytracer.camera_keys_pressed.get(key_name, False):
                self.raytracer.set_camera_key_state(key_name, True)
            event.accept()
            return
        
        # Dimension locking for object dragging
        if key == Qt.Key_X:
            self.dimension_locks['x'] = not self.dimension_locks['x']
            self.raytracer.set_dimension_lock('x', self.dimension_locks['x'])
            self.update_lock_status()
            event.accept()
        
        elif key == Qt.Key_Y:
            self.dimension_locks['y'] = not self.dimension_locks['y']
            self.raytracer.set_dimension_lock('y', self.dimension_locks['y'])
            self.update_lock_status()
            event.accept()
        
        elif key == Qt.Key_Z:
            self.dimension_locks['z'] = not self.dimension_locks['z']
            self.raytracer.set_dimension_lock('z', self.dimension_locks['z'])
            self.update_lock_status()
            event.accept()
        
        # Escape key to cancel operations
        elif key == Qt.Key_Escape:
            if self.dragging_object:
                self.raytracer.stop_object_dragging()
                self.dragging_object = False
                self.dimension_locks = {'x': False, 'y': False, 'z': False}
                self.update_lock_status()
                
                # Return to ray tracing if not manually in another mode
                if not self.manual_mode_change:
                    self.on_raytrace_mode()
            elif self.raytracer.camera_rotating:
                self.raytracer.stop_camera_rotation()
                if not self.manual_mode_change:
                    self.on_raytrace_mode()
            
            event.accept()
        
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release with better state management"""
        key = event.key()
        
        if key in self.camera_keys:
            key_name = self.camera_keys[key]
            if self.raytracer.camera_keys_pressed.get(key_name, False):
                self.raytracer.set_camera_key_state(key_name, False)
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    def update_lock_status(self):
        """Update lock status display"""
        locks = []
        for dim, locked in self.dimension_locks.items():
            if locked:
                locks.append(dim.upper())
        
        if locks:
            self.lock_label.setText(f"Locks: {', '.join(locks)}")
            self.lock_label.setStyleSheet("color: #ff9900; font-weight: bold;")
        else:
            self.lock_label.setText("Locks: None")
            self.lock_label.setStyleSheet("color: #888;")
    
    def get_lock_string(self):
        """Get string representation of active locks"""
        locks = [dim.upper() for dim, locked in self.dimension_locks.items() if locked]
        return ', '.join(locks) if locks else "None"
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.render_thread:
            self.render_thread.stop()
        self.raytracer.stop_rendering()
        event.accept()

