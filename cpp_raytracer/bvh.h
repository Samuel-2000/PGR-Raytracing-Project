#pragma once
#include "raytracer_core.h"
#include <algorithm>
#include <vector>

class AABB {
public:
    Vector3 min;
    Vector3 max;
    
    AABB() : min(Vector3(0,0,0)), max(Vector3(0,0,0)) {}
    AABB(const Vector3& a, const Vector3& b) : min(a), max(b) {}
    
    bool hit(const Ray& ray, double tmin, double tmax) const;
    static AABB surrounding_box(const AABB& box0, const AABB& box1);
};

AABB sphere_bounding_box(const Sphere& sphere);

class BVHNode {
public:
    AABB box;
    BVHNode* left;
    BVHNode* right;
    std::vector<Sphere> spheres;  // CHANGED: Store multiple spheres
    bool is_leaf;
    
    BVHNode();
    ~BVHNode();
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
};

class BVH {
private:
    BVHNode* root;
    
    BVHNode* build_tree(std::vector<Sphere>& spheres, size_t start, size_t end, int depth = 0);
    bool box_compare(const Sphere& a, const Sphere& b, int axis);
    
public:
    BVH();
    ~BVH();
    
    void build(const std::vector<Sphere>& spheres);
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
};