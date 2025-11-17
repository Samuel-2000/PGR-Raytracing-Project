# FILE: src/accelerators/bvh.py
"""
BVH Accelerator with SAH (Surface Area Heuristic)
"""
import numpy as np
from typing import List, Tuple, Optional
from core.scene import SceneObject, Ray, HitRecord

class BVHNode:
    __slots__ = ['bounds', 'left', 'right', 'object', 'is_leaf']
    
    def __init__(self):
        self.bounds = None
        self.left = None
        self.right = None
        self.object = None
        self.is_leaf = False

class BVHAccelerator:
    """BVH with Surface Area Heuristic for optimal splitting"""
    
    def __init__(self, max_objects_per_leaf: int = 4):
        self.root = None
        self.max_objects_per_leaf = max_objects_per_leaf
        self.nodes = 0
    
    def build(self, objects: List[SceneObject]):
        """Build BVH using SAH"""
        if not objects:
            self.root = None
            return
        
        self.nodes = 0
        self.root = self._build_node(objects)
    
    def _build_node(self, objects: List[SceneObject]) -> BVHNode:
        node = BVHNode()
        self.nodes += 1
        
        # Compute bounds for all objects
        node.bounds = self._compute_bounds(objects)
        
        if len(objects) <= self.max_objects_per_leaf:
            node.object = objects[0] if len(objects) == 1 else objects
            node.is_leaf = True
            return node
        
        # Find best split using SAH
        best_axis, best_split, best_cost = self._find_best_split(objects, node.bounds)
        
        if best_cost >= len(objects) * self._compute_surface_area(node.bounds):
            # Don't split, create leaf
            node.object = objects
            node.is_leaf = True
            return node
        
        # Split objects
        left_objs, right_objs = self._split_objects(objects, best_axis, best_split)
        
        node.left = self._build_node(left_objs)
        node.right = self._build_node(right_objs)
        node.is_leaf = False
        
        return node
    
    def _find_best_split(self, objects: List[SceneObject], bounds: Tuple) -> Tuple[int, float, float]:
        best_axis = 0
        best_split = 0
        best_cost = float('inf')
        
        bounds_min, bounds_max = bounds
        extent = bounds_max - bounds_min
        
        # Try all three axes
        for axis in range(3):
            if extent[axis] < 1e-8:
                continue
            
            # Sort objects by center along this axis
            sorted_objs = sorted(objects, 
                               key=lambda obj: self._get_object_center(obj)[axis])
            
            # Try different split positions
            for i in range(1, len(sorted_objs)):
                left_objs = sorted_objs[:i]
                right_objs = sorted_objs[i:]
                
                left_bounds = self._compute_bounds(left_objs)
                right_bounds = self._compute_bounds(right_objs)
                
                left_area = self._compute_surface_area(left_bounds)
                right_area = self._compute_surface_area(right_bounds)
                
                cost = (left_area * len(left_objs) + 
                       right_area * len(right_objs))
                
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_split = (self._get_object_center(sorted_objs[i-1])[axis] +
                                 self._get_object_center(sorted_objs[i])[axis]) / 2
        
        return best_axis, best_split, best_cost
    
    def _compute_bounds(self, objects: List[SceneObject]) -> Tuple:
        if not objects:
            return (np.zeros(3), np.zeros(3))
        
        all_mins = []
        all_maxs = []
        
        for obj in objects:
            bmin, bmax = obj.get_bounds()
            all_mins.append(bmin)
            all_maxs.append(bmax)
        
        scene_min = np.min(all_mins, axis=0)
        scene_max = np.max(all_maxs, axis=0)
        return (scene_min, scene_max)
    
    def _compute_surface_area(self, bounds: Tuple) -> float:
        min_bound, max_bound = bounds
        extent = max_bound - min_bound
        return 2.0 * (extent[0] * extent[1] + extent[0] * extent[2] + extent[1] * extent[2])
    
    def _get_object_center(self, obj: SceneObject) -> np.ndarray:
        bmin, bmax = obj.get_bounds()
        return (bmin + bmax) / 2
    
    def _split_objects(self, objects: List[SceneObject], axis: int, split_pos: float):
        left = []
        right = []
        
        for obj in objects:
            center = self._get_object_center(obj)
            if center[axis] < split_pos:
                left.append(obj)
            else:
                right.append(obj)
        
        # Ensure both sides have objects
        if not left or not right:
            mid = len(objects) // 2
            left = objects[:mid]
            right = objects[mid:]
        
        return left, right
    
    def intersect(self, ray: Ray) -> Optional[HitRecord]:
        """Intersect ray with BVH"""
        if self.root is None:
            return None
        return self._intersect_node(ray, self.root)
    
    def _intersect_node(self, ray: Ray, node: BVHNode) -> Optional[HitRecord]:
        # Check ray against node bounds
        if not self._intersect_aabb(ray, node.bounds):
            return None
        
        if node.is_leaf:
            return self._intersect_objects(ray, node.object)
        
        # Check both children
        hit_left = self._intersect_node(ray, node.left)
        hit_right = self._intersect_node(ray, node.right)
        
        if hit_left and hit_right:
            return hit_left if hit_left.distance < hit_right.distance else hit_right
        return hit_left or hit_right
    
    def _intersect_objects(self, ray: Ray, objects) -> Optional[HitRecord]:
        if isinstance(objects, list):
            closest_hit = None
            min_dist = float('inf')
            
            for obj in objects:
                hit = obj.intersect(ray)
                if hit and hit.distance < min_dist:
                    closest_hit = hit
                    min_dist = hit.distance
            
            return closest_hit
        else:
            return objects.intersect(ray)
    
    def _intersect_aabb(self, ray: Ray, bounds: Tuple) -> bool:
        if bounds is None:
            return False
            
        bmin, bmax = bounds
        
        tmin = 0.0
        tmax = float('inf')
        
        for i in range(3):
            if abs(ray.direction[i]) < 1e-8:
                # Ray parallel to slab
                if ray.origin[i] < bmin[i] or ray.origin[i] > bmax[i]:
                    return False
            else:
                inv_d = 1.0 / ray.direction[i]
                t1 = (bmin[i] - ray.origin[i]) * inv_d
                t2 = (bmax[i] - ray.origin[i]) * inv_d
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                
                if tmin > tmax:
                    return False
        
        return True
    
    def intersect_batch(self, rays: np.ndarray) -> List[Optional[HitRecord]]:
        """Batch intersection for multiple rays"""
        results = []
        for i in range(rays.shape[0]):
            ray = Ray(rays[i, 0], rays[i, 1])
            results.append(self.intersect(ray))
        return results