# FILE: src/core/scene.py
"""
Scene management with optimized object storage
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .materials import Material

@dataclass
class Ray:
    origin: np.ndarray  # [3]
    direction: np.ndarray  # [3]
    
    def at(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t

@dataclass 
class HitRecord:
    distance: float
    point: np.ndarray  # [3]
    normal: np.ndarray  # [3]
    material: Material
    object_id: int
    uv: np.ndarray = None  # [2]

class SceneObject:
    """Base class for all scene objects"""
    def __init__(self, material: Material, object_id: int):
        self.material = material
        self.object_id = object_id
        self.bounds = None
    
    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        raise NotImplementedError
    
    def get_bounds(self):
        raise NotImplementedError

class Sphere(SceneObject):
    """Sphere object with optimized intersection"""
    __slots__ = ['center', 'radius', 'material', 'object_id', 'bounds']
    
    def __init__(self, center: np.ndarray, radius: float, material: Material, object_id: int):
        super().__init__(material, object_id)
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self._compute_bounds()
    
    def _compute_bounds(self):
        r_vec = np.array([self.radius, self.radius, self.radius])
        self.bounds = (self.center - r_vec, self.center + r_vec)
    
    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        
        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        
        t = t1 if t1 > 1e-4 else (t2 if t2 > 1e-4 else None)
        if t is None or t > 1e10:
            return None
        
        point = ray.at(t)
        normal = (point - self.center) / self.radius
        normal = normal / np.linalg.norm(normal)
        
        return HitRecord(t, point, normal, self.material, self.object_id)
    
    def get_bounds(self):
        return self.bounds

class TriangleMesh(SceneObject):
    """Triangle mesh with efficient storage"""
    __slots__ = ['vertices', 'triangles', 'material', 'object_id', 'bounds', 'normals']
    
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, 
                 material: Material, object_id: int):
        super().__init__(material, object_id)
        self.vertices = np.array(vertices, dtype=np.float32)
        self.triangles = np.array(triangles, dtype=np.int32)
        self.normals = self._compute_normals()
        self._compute_bounds()
    
    def _compute_normals(self):
        normals = []
        for tri in self.triangles:
            v0, v1, v2 = self.vertices[tri]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
        return np.array(normals, dtype=np.float32)
    
    def _compute_bounds(self):
        min_bound = np.min(self.vertices, axis=0)
        max_bound = np.max(self.vertices, axis=0)
        self.bounds = (min_bound, max_bound)
    
    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        closest_hit = None
        min_t = float('inf')
        
        for i, tri in enumerate(self.triangles):
            hit = self._intersect_triangle(ray, tri, i)
            if hit and hit.distance < min_t:
                closest_hit = hit
                min_t = hit.distance
        
        return closest_hit
    
    def _intersect_triangle(self, ray: Ray, triangle: np.ndarray, tri_index: int) -> Optional[HitRecord]:
        # Möller–Trumbore intersection algorithm
        v0, v1, v2 = self.vertices[triangle]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray.direction, edge2)
        
        a = np.dot(edge1, h)
        if abs(a) < 1e-8:
            return None
        
        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        
        t = f * np.dot(edge2, q)
        if t > 1e-4 and t < 1e10:
            point = ray.at(t)
            normal = self.normals[tri_index]
            
            # Compute UV coordinates (barycentric)
            w = 1.0 - u - v
            uv = np.array([u, v])
            
            return HitRecord(t, point, normal, self.material, self.object_id, uv)
        
        return None

class Scene:
    """Scene container with efficient object management"""
    def __init__(self):
        self.objects: List[SceneObject] = []
        self.lights: List[SceneObject] = []
        self.background_color = np.array([0.1, 0.1, 0.2], dtype=np.float32)
        self.environment_map = None
        self._object_counter = 0
    
    def add_object(self, obj: SceneObject) -> int:
        obj.object_id = self._object_counter
        self.objects.append(obj)
        
        # Check if object is a light source
        if hasattr(obj.material, 'emission') and np.any(obj.material.emission > 0):
            self.lights.append(obj)
        
        self._object_counter += 1
        return obj.object_id
    
    def add_sphere(self, center, radius, material) -> int:
        sphere = Sphere(center, radius, material, self._object_counter)
        return self.add_object(sphere)
    
    def add_triangle_mesh(self, vertices, triangles, material) -> int:
        mesh = TriangleMesh(vertices, triangles, material, self._object_counter)
        return self.add_object(mesh)
    
    def get_bounds(self):
        if not self.objects:
            return (np.zeros(3), np.zeros(3))
        
        all_mins = []
        all_maxs = []
        
        for obj in self.objects:
            bmin, bmax = obj.get_bounds()
            all_mins.append(bmin)
            all_maxs.append(bmax)
        
        scene_min = np.min(all_mins, axis=0)
        scene_max = np.max(all_maxs, axis=0)
        return scene_min, scene_max