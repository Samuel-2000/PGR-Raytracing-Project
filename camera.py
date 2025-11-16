import numpy as np
from typing import Tuple

class Camera:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.target = np.array([0.0, 0.0, -1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.fov = 60.0  # Field of view in degrees
        self.aspect_ratio = 16.0 / 9.0
    
    def get_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ray for given screen coordinates"""
        # Convert from screen coordinates to world coordinates
        theta = np.radians(self.fov)
        half_height = np.tan(theta / 2)
        half_width = self.aspect_ratio * half_height
        
        # Camera basis vectors
        w = (self.position - self.target) / np.linalg.norm(self.position - self.target)
        u_vec = np.cross(self.up, w)
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = np.cross(w, u_vec)
        
        # Ray direction
        direction = (u - 0.5) * 2 * half_width * u_vec + \
                   (v - 0.5) * 2 * half_height * v_vec - w
        direction = direction / np.linalg.norm(direction)
        
        return self.position.copy(), direction
    
    def set_position(self, x: float, y: float, z: float):
        """Set camera position"""
        self.position = np.array([x, y, z])
    
    def look_at(self, target_x: float, target_y: float, target_z: float):
        """Set camera look-at target"""
        self.target = np.array([target_x, target_y, target_z])