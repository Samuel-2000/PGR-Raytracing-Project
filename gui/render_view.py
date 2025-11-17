# FILE: src/gui/render_view.py
"""
Interactive render view with zoom and pan
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QMouseEvent

class RenderView(QWidget):
    """Widget for displaying rendered images with interaction"""
    
    image_clicked = pyqtSignal(float, float)  # x, y in image coordinates
    
    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        self.current_image = None
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        
        self.setMouseTracking(True)
    
    def update_image(self, image: np.ndarray):
        """Update the displayed image"""
        self.current_image = image
        self._update_display()
    
    def _update_display(self):
        """Update the display with current image and transformations"""
        if self.current_image is None:
            return
        
        # Convert numpy array to QImage
        height, width = self.current_image.shape[:2]
        bytes_per_line = 3 * width
        
        # Convert to 8-bit
        image_8bit = (np.clip(self.current_image, 0, 1) * 255).astype(np.uint8)
        
        # Create QImage (assuming RGB format)
        q_image = QImage(image_8bit.data, width, height, bytes_per_line, 
                        QImage.Format_RGB888)
        
        # Apply zoom and pan
        if self.zoom_factor != 1.0 or any(self.pan_offset):
            pixmap = QPixmap.fromImage(q_image)
            
            # Calculate scaled size
            scaled_width = int(width * self.zoom_factor)
            scaled_height = int(height * self.zoom_factor)
            
            # Create scaled pixmap
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, 
                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_in = event.angleDelta().y() > 0
        old_zoom = self.zoom_factor
        
        if zoom_in:
            self.zoom_factor *= 1.2
        else:
            self.zoom_factor /= 1.2
        
        # Clamp zoom
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        
        if self.zoom_factor != old_zoom:
            self._update_display()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse clicks"""
        if event.button() == Qt.LeftButton and self.current_image is not None:
            # Convert to image coordinates
            img_width = self.current_image.shape[1]
            img_height = self.current_image.shape[0]
            
            label_size = self.image_label.size()
            x_scale = label_size.width() / img_width
            y_scale = label_size.height() / img_height
            scale = min(x_scale, y_scale)
            
            # Calculate offset for centered image
            x_offset = (label_size.width() - img_width * scale) / 2
            y_offset = (label_size.height() - img_height * scale) / 2
            
            # Convert click position to image coordinates
            img_x = (event.pos().x() - x_offset) / scale
            img_y = (event.pos().y() - y_offset) / scale
            
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.image_clicked.emit(img_x / img_width, img_y / img_height)
    
    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self._update_display()