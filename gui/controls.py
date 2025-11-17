# FILE: src/gui/controls.py
"""
Control panel for ray tracer settings
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QLabel, QPushButton, QSlider, QComboBox, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    """Control panel for ray tracer settings"""
    
    settings_changed = pyqtSignal(str, object)  # setting_name, value
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_signals()
    
    def setup_ui(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for organized settings
        tab_widget = QTabWidget()
        
        # Rendering tab
        render_tab = self.create_render_tab()
        tab_widget.addTab(render_tab, "Rendering")
        
        # Scene tab
        scene_tab = self.create_scene_tab()
        tab_widget.addTab(scene_tab, "Scene")
        
        # Materials tab
        materials_tab = self.create_materials_tab()
        tab_widget.addTab(materials_tab, "Materials")
        
        # Denoising tab
        denoise_tab = self.create_denoise_tab()
        tab_widget.addTab(denoise_tab, "Denoising")
        
        layout.addWidget(tab_widget)
        
        # Render controls
        render_controls = QGroupBox("Render Controls")
        render_layout = QHBoxLayout(render_controls)
        
        self.render_button = QPushButton("Start Rendering")
        self.render_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Image")
        
        render_layout.addWidget(self.render_button)
        render_layout.addWidget(self.stop_button)
        render_layout.addWidget(self.save_button)
        
        layout.addWidget(render_controls)
    
    def create_render_tab(self):
        """Create rendering settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Quality settings
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QVBoxLayout(quality_group)
        
        # Max samples
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Max Samples:"))
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 10000)
        self.max_samples.setValue(1024)
        samples_layout.addWidget(self.max_samples)
        quality_layout.addLayout(samples_layout)
        
        # Max bounces
        bounces_layout = QHBoxLayout()
        bounces_layout.addWidget(QLabel("Max Bounces:"))
        self.max_bounces = QSpinBox()
        self.max_bounces.setRange(1, 32)
        self.max_bounces.setValue(8)
        bounces_layout.addWidget(self.max_bounces)
        quality_layout.addLayout(bounces_layout)
        
        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["400x300", "800x600", "1200x900", "1600x1200"])
        self.resolution_combo.setCurrentText("800x600")
        res_layout.addWidget(self.resolution_combo)
        quality_layout.addLayout(res_layout)
        
        layout.addWidget(quality_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout(perf_group)
        
        # GPU acceleration
        self.gpu_checkbox = QCheckBox("GPU Acceleration")
        self.gpu_checkbox.setChecked(True)
        perf_layout.addWidget(self.gpu_checkbox)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Samples/Batch:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(16)
        batch_layout.addWidget(self.batch_size)
        perf_layout.addLayout(batch_layout)
        
        layout.addWidget(perf_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.importance_sampling = QCheckBox("Importance Sampling")
        self.importance_sampling.setChecked(True)
        advanced_layout.addWidget(self.importance_sampling)
        
        self.next_event_estimation = QCheckBox("Next Event Estimation")
        self.next_event_estimation.setChecked(True)
        advanced_layout.addWidget(self.next_event_estimation)
        
        self.russian_roulette = QCheckBox("Russian Roulette")
        self.russian_roulette.setChecked(True)
        advanced_layout.addWidget(self.russian_roulette)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        return widget
    
    def create_scene_tab(self):
        """Create scene settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Object selection
        obj_group = QGroupBox("Objects")
        obj_layout = QVBoxLayout(obj_group)
        
        self.object_combo = QComboBox()
        self.object_combo.addItems(["Sphere 1", "Sphere 2", "Sphere 3", "Ground", "Light"])
        obj_layout.addWidget(self.object_combo)
        
        # Object transform
        transform_group = QGroupBox("Transform")
        transform_layout = QVBoxLayout(transform_group)
        
        # Position controls
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Position:"))
        self.pos_x = QDoubleSpinBox()
        self.pos_y = QDoubleSpinBox()
        self.pos_z = QDoubleSpinBox()
        for spinbox in [self.pos_x, self.pos_y, self.pos_z]:
            spinbox.setRange(-100, 100)
            spinbox.setSingleStep(0.1)
            pos_layout.addWidget(spinbox)
        transform_layout.addLayout(pos_layout)
        
        obj_layout.addWidget(transform_group)
        layout.addWidget(obj_group)
        
        # Camera controls
        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_group)
        
        # Camera position
        cam_pos_layout = QHBoxLayout()
        cam_pos_layout.addWidget(QLabel("Camera Pos:"))
        self.cam_x = QDoubleSpinBox()
        self.cam_y = QDoubleSpinBox()
        self.cam_z = QDoubleSpinBox()
        for spinbox in [self.cam_x, self.cam_y, self.cam_z]:
            spinbox.setRange(-10, 10)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0.0)
            cam_pos_layout.addWidget(spinbox)
        cam_layout.addLayout(cam_pos_layout)
        
        layout.addWidget(cam_group)
        
        layout.addStretch()
        return widget
    
    def create_materials_tab(self):
        """Create materials settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Material properties
        mat_group = QGroupBox("Material Properties")
        mat_layout = QVBoxLayout(mat_group)
        
        # Albedo
        albedo_layout = QHBoxLayout()
        albedo_layout.addWidget(QLabel("Albedo:"))
        self.albedo_r = QDoubleSpinBox()
        self.albedo_g = QDoubleSpinBox()
        self.albedo_b = QDoubleSpinBox()
        for spinbox in [self.albedo_r, self.albedo_g, self.albedo_b]:
            spinbox.setRange(0, 1)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(0.8)
            albedo_layout.addWidget(spinbox)
        mat_layout.addLayout(albedo_layout)
        
        # Metallic
        metallic_layout = QHBoxLayout()
        metallic_layout.addWidget(QLabel("Metallic:"))
        self.metallic_slider = QSlider(Qt.Horizontal)
        self.metallic_slider.setRange(0, 100)
        self.metallic_value = QLabel("0.0")
        metallic_layout.addWidget(self.metallic_slider)
        metallic_layout.addWidget(self.metallic_value)
        mat_layout.addLayout(metallic_layout)
        
        # Roughness
        roughness_layout = QHBoxLayout()
        roughness_layout.addWidget(QLabel("Roughness:"))
        self.roughness_slider = QSlider(Qt.Horizontal)
        self.roughness_slider.setRange(0, 100)
        self.roughness_value = QLabel("0.5")
        roughness_layout.addWidget(self.roughness_slider)
        roughness_layout.addWidget(self.roughness_value)
        mat_layout.addLayout(roughness_layout)
        
        # Emission
        emission_layout = QHBoxLayout()
        emission_layout.addWidget(QLabel("Emission:"))
        self.emission_r = QDoubleSpinBox()
        self.emission_g = QDoubleSpinBox()
        self.emission_b = QDoubleSpinBox()
        for spinbox in [self.emission_r, self.emission_g, self.emission_b]:
            spinbox.setRange(0, 100)
            spinbox.setSingleStep(1.0)
            spinbox.setValue(0.0)
            emission_layout.addWidget(spinbox)
        mat_layout.addLayout(emission_layout)
        
        layout.addWidget(mat_group)
        layout.addStretch()
        return widget
    
    def create_denoise_tab(self):
        """Create denoising settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Denoiser selection
        denoise_group = QGroupBox("Denoiser")
        denoise_layout = QVBoxLayout(denoise_group)
        
        self.denoiser_combo = QComboBox()
        self.denoiser_combo.addItems(["None", "Bilateral", "NL-Means", "AI Denoiser"])
        denoise_layout.addWidget(self.denoiser_combo)
        
        # Denoiser parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Strength
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.denoise_strength = QSlider(Qt.Horizontal)
        self.denoise_strength.setRange(1, 100)
        self.denoise_strength.setValue(50)
        strength_layout.addWidget(self.denoise_strength)
        params_layout.addLayout(strength_layout)
        
        denoise_layout.addWidget(params_group)
        layout.addWidget(denoise_group)
        
        # Post-processing
        post_group = QGroupBox("Post-Processing")
        post_layout = QVBoxLayout(post_group)
        
        self.tone_mapping = QCheckBox("Tone Mapping (ACES)")
        self.tone_mapping.setChecked(True)
        post_layout.addWidget(self.tone_mapping)
        
        self.gamma_correction = QCheckBox("Gamma Correction")
        self.gamma_correction.setChecked(True)
        post_layout.addWidget(self.gamma_correction)
        
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Exposure:"))
        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(0.1, 5.0)
        self.exposure.setSingleStep(0.1)
        self.exposure.setValue(1.0)
        exposure_layout.addWidget(self.exposure)
        post_layout.addLayout(exposure_layout)
        
        layout.addWidget(post_group)
        layout.addStretch()
        return widget
    
    def setup_signals(self):
        """Connect control signals"""
        # Rendering settings
        self.max_samples.valueChanged.connect(
            lambda v: self.settings_changed.emit('max_samples', v))
        self.max_bounces.valueChanged.connect(
            lambda v: self.settings_changed.emit('max_bounces', v))
        
        # Material settings
        self.metallic_slider.valueChanged.connect(self.on_metallic_changed)
        self.roughness_slider.valueChanged.connect(self.on_roughness_changed)
    
    def on_metallic_changed(self, value):
        """Handle metallic slider change"""
        metallic = value / 100.0
        self.metallic_value.setText(f"{metallic:.2f}")
        self.settings_changed.emit('metallic', metallic)
    
    def on_roughness_changed(self, value):
        """Handle roughness slider change"""
        roughness = value / 100.0
        self.roughness_value.setText(f"{roughness:.2f}")
        self.settings_changed.emit('roughness', roughness)