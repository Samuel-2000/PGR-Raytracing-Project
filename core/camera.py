# FILE: camera.py
"""
Advanced Camera System with Multiple Projection Types and Physical Properties
"""
import numpy as np
import math
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def length_squared(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def normalize(self):
        length = self.length()
        if length > 1e-8:
            return self * (1.0 / length)
        return Vector3(0, 0, 1)
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr):
        return Vector3(arr[0], arr[1], arr[2])

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()
    
    def at(self, t):
        return self.origin + self.direction * t

class Camera:
    """
    Advanced physically-based camera with multiple projection types
    and realistic camera properties.
    """
    
    def __init__(self,
                 position: Vector3 = None,
                 target: Vector3 = None,
                 up: Vector3 = None,
                 fov: float = 45.0,
                 aspect_ratio: float = 16.0/9.0,
                 aperture: float = 0.0,
                 focus_distance: float = 1.0,
                 projection_type: str = "perspective",
                 near_plane: float = 0.1,
                 far_plane: float = 100.0,
                 exposure: float = 1.0,
                 shutter_speed: float = 1.0/60.0,
                 iso: float = 100.0):
        
        # Camera transform
        self.position = position if position else Vector3(0, 0, 0)
        self.target = target if target else Vector3(0, 0, -1)
        self.up = up if up else Vector3(0, 1, 0)
        
        # Camera properties
        self.fov = fov  # Vertical field of view in degrees
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture  # Lens aperture (for depth of field)
        self.focus_distance = focus_distance
        self.projection_type = projection_type  # "perspective", "orthographic", "fisheye"
        
        # Clipping planes
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        # Physical camera properties
        self.exposure = exposure
        self.shutter_speed = shutter_speed
        self.iso = iso
        
        # Camera coordinate system (cached)
        self._camera_to_world = None
        self._world_to_camera = None
        self._update_coordinate_system()
        
        # Precomputed values
        self._tan_half_fov = math.tan(math.radians(self.fov / 2))
        self._lens_radius = self.aperture / 2
        
        # Motion blur (for temporal effects)
        self.velocity = Vector3(0, 0, 0)
        self.angular_velocity = Vector3(0, 0, 0)
        
        # Depth of field
        self.focal_length = 50.0  # mm
        self.sensor_size = 36.0   # mm (full frame)
    
    def _update_coordinate_system(self):
        """Update camera coordinate system based on position, target, and up"""
        # Forward vector (points towards target)
        self.forward = (self.target - self.position).normalize()
        
        # Right vector
        self.right = self.forward.cross(self.up).normalize()
        
        # Recompute up vector to ensure orthonormal basis
        self.up = self.right.cross(self.forward).normalize()
        
        # Build camera to world matrix
        self._camera_to_world = np.eye(4)
        self._camera_to_world[:3, 0] = self.right.to_array()    # X axis
        self._camera_to_world[:3, 1] = self.up.to_array()       # Y axis  
        self._camera_to_world[:3, 2] = -self.forward.to_array() # Z axis (negative forward)
        self._camera_to_world[:3, 3] = self.position.to_array() # Position
        
        # World to camera is inverse of camera to world
        self._world_to_camera = np.linalg.inv(self._camera_to_world)
    
    def set_position(self, position: Vector3):
        """Set camera position and update coordinate system"""
        self.position = position
        self._update_coordinate_system()
    
    def set_target(self, target: Vector3):
        """Set camera target and update coordinate system"""
        self.target = target
        self._update_coordinate_system()
    
    def set_fov(self, fov: float):
        """Set field of view and update precomputed values"""
        self.fov = max(1.0, min(179.0, fov))
        self._tan_half_fov = math.tan(math.radians(self.fov / 2))
    
    def set_aperture(self, aperture: float):
        """Set lens aperture for depth of field"""
        self.aperture = max(0.0, aperture)
        self._lens_radius = self.aperture / 2
    
    def look_at(self, position: Vector3, target: Vector3, up: Vector3 = None):
        """Convenience method to set camera look-at parameters"""
        self.position = position
        self.target = target
        if up:
            self.up = up
        self._update_coordinate_system()
    
    def move(self, delta: Vector3):
        """Move camera by delta vector"""
        self.position = self.position + delta
        self.target = self.target + delta
        self._update_coordinate_system()
    
    def rotate_around_target(self, yaw: float, pitch: float, roll: float = 0.0):
        """Rotate camera around target point"""
        # Convert to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # Vector from target to camera
        camera_offset = self.position - self.target
        
        # Yaw rotation (around world up axis)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        yaw_rotated = Vector3(
            camera_offset.x * cos_yaw - camera_offset.z * sin_yaw,
            camera_offset.y,
            camera_offset.x * sin_yaw + camera_offset.z * cos_yaw
        )
        
        # Pitch rotation (around right axis)
        right = yaw_rotated.cross(self.up).normalize()
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        
        # Rotate around right axis
        pitch_rotated = Vector3(
            yaw_rotated.x * cos_pitch + yaw_rotated.y * sin_pitch,
            -yaw_rotated.x * sin_pitch + yaw_rotated.y * cos_pitch,
            yaw_rotated.z
        )
        
        # Update position
        self.position = self.target + pitch_rotated
        self._update_coordinate_system()
    
    def get_ray(self, u: float, v: float, time: float = 0.0) -> Ray:
        """
        Generate ray for given normalized screen coordinates (0-1)
        
        Args:
            u: Horizontal coordinate (0 = left, 1 = right)
            v: Vertical coordinate (0 = bottom, 1 = top)
            time: Time for motion blur (0-1)
            
        Returns:
            Ray: Camera ray for the given coordinates
        """
        if self.projection_type == "orthographic":
            return self._get_orthographic_ray(u, v, time)
        elif self.projection_type == "fisheye":
            return self._get_fisheye_ray(u, v, time)
        else:  # perspective
            return self._get_perspective_ray(u, v, time)
    
    def _get_perspective_ray(self, u: float, v: float, time: float) -> Ray:
        """Generate perspective projection ray"""
        # Convert to screen coordinates (-1 to 1)
        screen_x = (2.0 * u - 1.0) * self._tan_half_fov * self.aspect_ratio
        screen_y = (1.0 - 2.0 * v) * self._tan_half_fov
        
        # Ray direction in camera space
        direction_camera = Vector3(screen_x, screen_y, -1.0).normalize()
        
        # Transform to world space
        direction_world = self._camera_to_world_transform(direction_camera)
        
        # Apply depth of field if aperture > 0
        if self._lens_radius > 0:
            return self._get_dof_ray(direction_world, time)
        else:
            # Simple perspective ray
            ray_origin = self._get_motion_blur_origin(time)
            return Ray(ray_origin, direction_world)
    
    def _get_orthographic_ray(self, u: float, v: float, time: float) -> Ray:
        """Generate orthographic projection ray"""
        # Convert to screen coordinates
        screen_x = (2.0 * u - 1.0) * self._tan_half_fov * self.aspect_ratio * self.focus_distance
        screen_y = (1.0 - 2.0 * v) * self._tan_half_fov * self.focus_distance
        
        # Ray origin in camera space (on near plane)
        origin_camera = Vector3(screen_x, screen_y, 0)
        
        # Transform to world space
        origin_world = self._camera_to_world_transform(origin_camera, is_point=True)
        direction_world = self.forward  # Always forward in orthographic
        
        return Ray(origin_world, direction_world)
    
    def _get_fisheye_ray(self, u: float, v: float, time: float) -> Ray:
        """Generate fisheye projection ray"""
        # Convert to normalized device coordinates (-1 to 1)
        ndc_x = 2.0 * u - 1.0
        ndc_y = 2.0 * v - 1.0
        
        # Calculate radius and angle
        r = math.sqrt(ndc_x * ndc_x + ndc_y * ndc_y)
        
        if r > 1.0:
            # Outside fisheye circle
            return Ray(self.position, Vector3(0, 0, -1))
        
        # Fisheye mapping (equidistant projection)
        theta = r * math.radians(self.fov) / 2
        phi = math.atan2(ndc_y, ndc_x)
        
        # Direction in camera space
        sin_theta = math.sin(theta)
        direction_camera = Vector3(
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            -math.cos(theta)
        )
        
        # Transform to world space
        direction_world = self._camera_to_world_transform(direction_camera)
        
        return Ray(self.position, direction_world)
    
    def _get_dof_ray(self, ideal_direction: Vector3, time: float) -> Ray:
        """Generate ray with depth of field effects"""
        # Calculate point on focal plane
        focal_point = self.position + ideal_direction * self.focus_distance
        
        # Sample point on lens
        lens_sample = self._sample_lens()
        lens_offset = self.right * lens_sample[0] + self.up * lens_sample[1]
        
        # Calculate new origin and direction
        ray_origin = self.position + lens_offset
        ray_direction = (focal_point - ray_origin).normalize()
        
        # Apply motion blur to origin
        ray_origin = self._apply_motion_blur(ray_origin, time)
        
        return Ray(ray_origin, ray_direction)
    
    def _sample_lens(self) -> Tuple[float, float]:
        """Sample point on lens for depth of field"""
        # Uniform disk sampling
        r = math.sqrt(np.random.random()) * self._lens_radius
        theta = 2 * math.pi * np.random.random()
        return (r * math.cos(theta), r * math.sin(theta))
    
    def _get_motion_blur_origin(self, time: float) -> Vector3:
        """Apply motion blur to camera origin"""
        return self._apply_motion_blur(self.position, time)
    
    def _apply_motion_blur(self, point: Vector3, time: float) -> Vector3:
        """Apply motion blur to a point based on camera velocity"""
        if time > 0 and self.velocity.length_squared() > 0:
            # Linear motion blur
            point = point + self.velocity * time
        return point
    
    def _camera_to_world_transform(self, vector: Vector3, is_point: bool = False) -> Vector3:
        """Transform vector from camera space to world space"""
        vec_array = vector.to_array()
        if is_point:
            # For points, use full transformation (include translation)
            vec_array = np.append(vec_array, 1.0)
            world_array = self._camera_to_world @ vec_array
            return Vector3.from_array(world_array[:3])
        else:
            # For vectors, use only rotation (no translation)
            vec_array = np.append(vec_array, 0.0)
            world_array = self._camera_to_world @ vec_array
            return Vector3.from_array(world_array[:3])
    
    def get_ray_batch(self, u: np.ndarray, v: np.ndarray, time: float = 0.0) -> List[Ray]:
        """
        Generate multiple rays for batch processing
        
        Args:
            u: Array of horizontal coordinates (0-1)
            v: Array of vertical coordinates (0-1)
            time: Time for motion blur
            
        Returns:
            List[Ray]: List of camera rays
        """
        rays = []
        for i in range(len(u)):
            rays.append(self.get_ray(u[i], v[i], time))
        return rays
    
    def get_frustum_corners(self, distance: float = 1.0) -> List[Vector3]:
        """Get frustum corners at specified distance from camera"""
        corners = []
        for u in [0.0, 1.0]:
            for v in [0.0, 1.0]:
                ray = self.get_ray(u, v)
                corner = ray.origin + ray.direction * distance
                corners.append(corner)
        return corners
    
    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix (world to camera transformation)"""
        return self._world_to_camera.copy()
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix based on camera type"""
        if self.projection_type == "orthographic":
            return self._get_orthographic_matrix()
        else:
            return self._get_perspective_matrix()
    
    def _get_perspective_matrix(self) -> np.ndarray:
        """Get perspective projection matrix"""
        f = 1.0 / math.tan(math.radians(self.fov) / 2)
        aspect = self.aspect_ratio
        
        near, far = self.near_plane, self.far_plane
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])
    
    def _get_orthographic_matrix(self) -> np.ndarray:
        """Get orthographic projection matrix"""
        top = self._tan_half_fov * self.focus_distance
        right = top * self.aspect_ratio
        left, bottom = -right, -top
        
        near, far = self.near_plane, self.far_plane
        
        return np.array([
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ])
    
    def calculate_exposure_value(self) -> float:
        """Calculate exposure value based on physical camera properties"""
        # Simplified exposure calculation
        ev = math.log2((self.aperture * self.aperture) / self.shutter_speed) + math.log2(self.iso / 100)
        return ev
    
    def auto_exposure(self, scene_luminance: float):
        """Automatically adjust exposure based on scene luminance"""
        target_luminance = 0.18  # Middle gray
        self.exposure = target_luminance / (scene_luminance + 1e-8)
    
    def clone(self) -> 'Camera':
        """Create a copy of this camera"""
        return Camera(
            position=Vector3(self.position.x, self.position.y, self.position.z),
            target=Vector3(self.target.x, self.target.y, self.target.z),
            up=Vector3(self.up.x, self.up.y, self.up.z),
            fov=self.fov,
            aspect_ratio=self.aspect_ratio,
            aperture=self.aperture,
            focus_distance=self.focus_distance,
            projection_type=self.projection_type,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            exposure=self.exposure,
            shutter_speed=self.shutter_speed,
            iso=self.iso
        )

class CameraAnimator:
    """Utility class for camera animation and movement"""
    
    def __init__(self, camera: Camera):
        self.camera = camera
        self.animation_path = []
        self.current_time = 0.0
        self.duration = 1.0
    
    def add_keyframe(self, position: Vector3, target: Vector3, time: float):
        """Add camera keyframe to animation path"""
        self.animation_path.append({
            'position': position,
            'target': target,
            'time': time
        })
        # Sort by time
        self.animation_path.sort(key=lambda x: x['time'])
    
    def update(self, dt: float):
        """Update camera animation"""
        if len(self.animation_path) < 2:
            return
        
        self.current_time += dt
        if self.current_time > self.duration:
            self.current_time = 0.0
        
        # Find current segment
        for i in range(len(self.animation_path) - 1):
            kf1 = self.animation_path[i]
            kf2 = self.animation_path[i + 1]
            
            if kf1['time'] <= self.current_time <= kf2['time']:
                # Interpolate
                t = (self.current_time - kf1['time']) / (kf2['time'] - kf1['time'])
                position = self._interpolate_position(kf1['position'], kf2['position'], t)
                target = self._interpolate_target(kf1['target'], kf2['target'], t)
                
                self.camera.look_at(position, target)
                break
    
    def _interpolate_position(self, p1: Vector3, p2: Vector3, t: float) -> Vector3:
        """Interpolate position with smooth curve"""
        # Cubic interpolation for smooth motion
        t2 = t * t
        t3 = t2 * t
        return p1 * (2*t3 - 3*t2 + 1) + p2 * (-2*t3 + 3*t2)
    
    def _interpolate_target(self, t1: Vector3, t2: Vector3, t: float) -> Vector3:
        """Interpolate target with spherical linear interpolation"""
        return t1 * (1 - t) + t2 * t

# Predefined camera configurations
class CameraPresets:
    """Factory for common camera configurations"""
    
    @staticmethod
    def default() -> Camera:
        return Camera(
            position=Vector3(0, 1, 3),
            target=Vector3(0, 0, -1),
            fov=45.0
        )
    
    @staticmethod
    def wide_angle() -> Camera:
        return Camera(
            position=Vector3(0, 1, 2),
            target=Vector3(0, 0, -1),
            fov=90.0
        )
    
    @staticmethod
    def telephoto() -> Camera:
        return Camera(
            position=Vector3(0, 1, 10),
            target=Vector3(0, 0, -1),
            fov=20.0
        )
    
    @staticmethod
    def top_down() -> Camera:
        return Camera(
            position=Vector3(0, 10, 0),
            target=Vector3(0, 0, 0),
            up=Vector3(0, 0, -1),
            fov=45.0
        )
    
    @staticmethod
    def cinematic_dof() -> Camera:
        return Camera(
            position=Vector3(0, 1, 5),
            target=Vector3(0, 0, 0),
            fov=45.0,
            aperture=0.1,
            focus_distance=5.0
        )
    
    @staticmethod
    def orthographic_top() -> Camera:
        return Camera(
            position=Vector3(0, 10, 0),
            target=Vector3(0, 0, 0),
            up=Vector3(0, 0, -1),
            fov=45.0,
            projection_type="orthographic"
        )