from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Material:
    albedo: Vector3 = None
    metallic: float = 0.0
    roughness: float = 0.5
    emission: Vector3 = None
    ior: float = 1.5
    
    def __post_init__(self):
        if self.albedo is None:
            self.albedo = Vector3(0.8, 0.8, 0.8)
        if self.emission is None:
            self.emission = Vector3(0.0, 0.0, 0.0)

@dataclass
class Sphere:
    center: Vector3 = None
    radius: float = 1.0
    material: Material = None
    object_id: int = 0
    
    def __post_init__(self):
        if self.center is None:
            self.center = Vector3(0, 0, 0)
        if self.material is None:
            self.material = Material()

class Scene:
    def __init__(self):
        self.spheres: List[Sphere] = []
        self.background_color = Vector3(0.1, 0.1, 0.1)
        self.use_bvh = True
    
    def add_sphere(self, sphere: Sphere):
        self.spheres.append(sphere)
    
    def build_bvh(self):
        """Build BVH for the scene - placeholder for Python"""
        pass

class SceneManager:
    def __init__(self):
        self.scenes: Dict[str, Scene] = {}
        self.current_scene: Union[Scene, None] = None
        self.object_counter = 0
    
    def create_complex_scene(self, num_spheres: int = 100) -> Scene:
        """Create a complex scene with many spheres to demonstrate BVH benefits"""
        scene = Scene()
        scene.use_bvh = True
        
        # Create a ground plane
        ground_material = Material(
            albedo=Vector3(0.8, 0.8, 0.8),
            metallic=0.0,
            roughness=0.9
        )
        ground = Sphere(
            center=Vector3(0, -100.5, -5),
            radius=100.0,
            material=ground_material,
            object_id=self._get_next_object_id()
        )
        scene.add_sphere(ground)
        
        # Create many random spheres
        import random
        for i in range(num_spheres):
            material = Material(
                albedo=Vector3(
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9), 
                    random.uniform(0.1, 0.9)
                ),
                metallic=random.uniform(0.0, 1.0),
                roughness=random.uniform(0.1, 0.9),
                ior=random.uniform(1.3, 2.4)
            )
            
            sphere = Sphere(
                center=Vector3(
                    random.uniform(-8, 8),
                    random.uniform(-2, 4),
                    random.uniform(-12, -2)
                ),
                radius=random.uniform(0.2, 1.0),
                material=material,
                object_id=self._get_next_object_id()
            )
            scene.add_sphere(sphere)
        
        # Light source
        light_material = Material(
            albedo=Vector3(1.0, 1.0, 0.9),
            emission=Vector3(4.0, 4.0, 3.5),
            metallic=0.0,
            roughness=0.1
        )
        light = Sphere(
            center=Vector3(0, 8, -5),
            radius=2.0,
            material=light_material,
            object_id=self._get_next_object_id()
        )
        scene.add_sphere(light)
        
        self.current_scene = scene
        return scene
    
    def create_pbr_demo_scene(self) -> Scene:
        """Create a demo scene showcasing PBR materials"""
        scene = Scene()
        scene.use_bvh = True
        
        # Ground
        ground_material = Material(
            albedo=Vector3(0.8, 0.8, 0.8),
            metallic=0.0,
            roughness=0.9
        )
        ground = Sphere(
            center=Vector3(0, -100.5, -5),
            radius=100.0,
            material=ground_material,
            object_id=self._get_next_object_id()
        )
        scene.add_sphere(ground)
        
        # Metallic spheres with varying roughness
        for i in range(3):
            material = Material(
                albedo=Vector3(0.8, 0.8, 0.8),
                metallic=0.9,
                roughness=i * 0.3 + 0.1,
                ior=1.5
            )
            sphere = Sphere(
                center=Vector3(i * 2.5 - 2.5, 0, -5),
                radius=0.8,
                material=material,
                object_id=self._get_next_object_id()
            )
            scene.add_sphere(sphere)
        
        # Dielectric spheres
        for i in range(3):
            material = Material(
                albedo=Vector3(0.2, 0.5, 0.8),
                metallic=0.0,
                roughness=i * 0.3 + 0.1,
                ior=1.3 + i * 0.2
            )
            sphere = Sphere(
                center=Vector3(i * 2.5 - 2.5, 2.5, -5),
                radius=0.8,
                material=material,
                object_id=self._get_next_object_id()
            )
            scene.add_sphere(sphere)
        
        # Emissive sphere
        emissive_material = Material(
            albedo=Vector3(0.9, 0.9, 0.5),
            emission=Vector3(2.0, 2.0, 1.0),
            metallic=0.0,
            roughness=0.2
        )
        emissive_sphere = Sphere(
            center=Vector3(0, -2, -3),
            radius=1.0,
            material=emissive_material,
            object_id=self._get_next_object_id()
        )
        scene.add_sphere(emissive_sphere)
        
        self.current_scene = scene
        return scene
    
    def move_object(self, object_id: int, new_position: tuple):
        """Move an object to a new position"""
        if self.current_scene:
            for sphere in self.current_scene.spheres:
                if sphere.object_id == object_id:
                    sphere.center = Vector3(*new_position)
                    break
    
    def toggle_bvh(self, use_bvh: bool):
        """Toggle BVH acceleration on/off"""
        if self.current_scene:
            self.current_scene.use_bvh = use_bvh
    
    def _get_next_object_id(self) -> int:
        self.object_counter += 1
        return self.object_counter