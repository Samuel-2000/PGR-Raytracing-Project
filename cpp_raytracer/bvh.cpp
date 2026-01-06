// bvh.cpp - FIXED VERSION
#include "bvh.h"
#include <algorithm>
#include <iostream>

bool AABB::hit(const Ray& ray, double tmin, double tmax) const {
    for (int a = 0; a < 3; a++) {
        double invD = 1.0 / ray.direction[a];
        double t0 = (min[a] - ray.origin[a]) * invD;
        double t1 = (max[a] - ray.origin[a]) * invD;
        if (invD < 0.0) {
            std::swap(t0, t1);
        }
        tmin = (t0 > tmin) ? t0 : tmin;
        tmax = (t1 < tmax) ? t1 : tmax;
        if (tmax <= tmin) {
            return false;
        }
    }
    return true;
}

AABB AABB::surrounding_box(const AABB& box0, const AABB& box1) {
    Vector3 small(
        std::fmin(box0.min.x, box1.min.x),
        std::fmin(box0.min.y, box1.min.y),
        std::fmin(box0.min.z, box1.min.z)
    );
    Vector3 big(
        std::fmax(box0.max.x, box1.max.x),
        std::fmax(box0.max.y, box1.max.y),
        std::fmax(box0.max.z, box1.max.z)
    );
    return AABB(small, big);
}

AABB sphere_bounding_box(const Sphere& sphere) {
    Vector3 radius_vec(sphere.radius, sphere.radius, sphere.radius);
    return AABB(sphere.center - radius_vec, sphere.center + radius_vec);
}

BVHNode::BVHNode() : left(nullptr), right(nullptr), is_leaf(false) {}

BVHNode::~BVHNode() {
    delete left;
    delete right;
}

bool BVHNode::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec, 
                 const std::vector<Sphere>& scene_spheres) const {
    if (!box.hit(ray, t_min, t_max)) {
        return false;
    }
    
    if (is_leaf) {
        bool hit_anything = false;
        HitRecord temp_rec;
        double closest_so_far = t_max;

        for (int sphere_idx : sphere_indices) {
            if (scene_spheres[sphere_idx].hit(ray, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
    
    // Internal node - check children
    HitRecord left_rec, right_rec;
    bool hit_left = (left != nullptr) && left->hit(ray, t_min, t_max, left_rec, scene_spheres);
    bool hit_right = (right != nullptr) && right->hit(ray, t_min, t_max, right_rec, scene_spheres);
    
    if (hit_left && hit_right) {
        rec = (left_rec.t < right_rec.t) ? left_rec : right_rec;
        return true;
    }
    else if (hit_left) {
        rec = left_rec;
        return true;
    }
    else if (hit_right) {
        rec = right_rec;
        return true;
    }
    
    return false;
}

bool BVH::box_compare(const Sphere& a, const Sphere& b, int axis) {
    AABB box_a = sphere_bounding_box(a);
    AABB box_b = sphere_bounding_box(b);
    
    // Use bounding box CENTERS, not minimum edges
    Vector3 center_a = box_a.center();
    Vector3 center_b = box_b.center();
    
    if (axis == 0) {
        return center_a.x < center_b.x;
    }
    else if (axis == 1) {
        return center_a.y < center_b.y;
    }
    else {
        return center_a.z < center_b.z;
    }
}

BVHNode* BVH::build_tree(const std::vector<Sphere>& scene_spheres, 
                        std::vector<int>& indices, size_t start, size_t end, 
                        int depth, bool debug_mode) {
    if (start >= end || indices.empty()) {
        if (debug_mode) {
            std::cout << "[BVH::build_tree] Invalid range: start=" << start 
                      << " end=" << end << std::endl;
        }
        return nullptr;
    }
    
    BVHNode* node = new BVHNode();
    node_count++;
    size_t span = end - start;
    
    if (debug_mode && depth <= 3) {  // Only show first few levels
        std::cout << "[BVH::build_tree] Depth=" << depth 
                  << " span=" << span 
                  << " indices=[" << start << "-" << end << ")" << std::endl;
    }
    
    // For small numbers of spheres, create a leaf node
    if (span <= 4) {
        for (size_t i = start; i < end; i++) {
            node->sphere_indices.push_back(indices[i]);
        }
        
        // Calculate bounding box for all spheres in this leaf
        if (!node->sphere_indices.empty()) {
            int first_idx = node->sphere_indices[0];
            node->box = sphere_bounding_box(scene_spheres[first_idx]);
            for (size_t i = 1; i < node->sphere_indices.size(); i++) {
                int idx = node->sphere_indices[i];
                node->box = AABB::surrounding_box(node->box, 
                                                 sphere_bounding_box(scene_spheres[idx]));
            }
        }
        node->is_leaf = true;
        
        if (debug_mode && depth <= 3) {
            std::cout << "[BVH::build_tree] Created LEAF node at depth=" << depth 
                      << " with " << node->sphere_indices.size() 
                      << " spheres: ";
            for (int idx : node->sphere_indices) {
                std::cout << idx << "(" << scene_spheres[idx].name << ") ";
            }
            std::cout << std::endl;
        }
        
        return node;
    }
    
    // Calculate total bounding box
    int first_idx = indices[start];
    AABB total_box = sphere_bounding_box(scene_spheres[first_idx]);
    for (size_t i = start + 1; i < end; i++) {
        int idx = indices[i];
        total_box = AABB::surrounding_box(total_box, 
                                         sphere_bounding_box(scene_spheres[idx]));
    }
    node->box = total_box;
    
    if (debug_mode && depth <= 2) {
        std::cout << "[BVH::build_tree] Internal node at depth=" << depth 
                  << " box.min=(" << total_box.min.x << "," << total_box.min.y << "," << total_box.min.z << ")"
                  << " box.max=(" << total_box.max.x << "," << total_box.max.y << "," << total_box.max.z << ")"
                  << std::endl;
    }
    
    // Choose split axis based on largest extent
    Vector3 extent = total_box.max - total_box.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.y && extent.z > extent.x) axis = 2;
    
    if (debug_mode && depth <= 2) {
        std::cout << "[BVH::build_tree] Split axis=" << axis 
                  << " extent=(" << extent.x << "," << extent.y << "," << extent.z << ")" << std::endl;
    }
    
    // Sort indices based on sphere positions along chosen axis
    auto comparator = [axis, &scene_spheres, this](int idx_a, int idx_b) {
        return this->box_compare(scene_spheres[idx_a], scene_spheres[idx_b], axis);
    };
    std::sort(indices.begin() + start, indices.begin() + end, comparator);
    
    // Split at midpoint
    size_t mid = start + span / 2;
    
    // Recursively build children
    node->left = build_tree(scene_spheres, indices, start, mid, depth + 1, debug_mode);
    node->right = build_tree(scene_spheres, indices, mid, end, depth + 1, debug_mode);
    node->is_leaf = false;
    
    return node;
}

BVH::BVH() : root(nullptr), node_count(0) {}

BVH::~BVH() {
    delete root;
}

void BVH::build(const std::vector<Sphere>& scene_spheres, bool debug_mode) {
    if (scene_spheres.empty()) {
        if (debug_mode) {
            std::cout << "[BVH::build] WARNING: No spheres to build BVH!" << std::endl;
        }
        return;
    }
    
    if (debug_mode) {
        std::cout << "\n[BVH::build] Starting build with " << scene_spheres.size() << " spheres:" << std::endl;
        for (size_t i = 0; i < scene_spheres.size(); i++) {
            const auto& s = scene_spheres[i];
            AABB bbox = sphere_bounding_box(s);
            std::cout << "  [" << i << "] " << s.name 
                      << " id=" << s.object_id
                      << " pos=(" << s.center.x << "," << s.center.y << "," << s.center.z << ")"
                      << " radius=" << s.radius
                      << " bbox_min=(" << bbox.min.x << "," << bbox.min.y << "," << bbox.min.z << ")"
                      << " bbox_max=(" << bbox.max.x << "," << bbox.max.y << "," << bbox.max.z << ")"
                      << std::endl;
        }
    }
    
    // Delete old tree if exists
    delete root;
    root = nullptr;
    node_count = 0;
    
    // Create indices vector
    std::vector<int> indices(scene_spheres.size());
    for (size_t i = 0; i < scene_spheres.size(); i++) {
        indices[i] = i;
    }
    
    if (debug_mode) {
        std::cout << "[BVH::build] Building tree with indices: ";
        for (int idx : indices) std::cout << idx << " ";
        std::cout << std::endl;
    }
    
    root = build_tree(scene_spheres, indices, 0, indices.size(), 0, debug_mode);
    
    if (debug_mode) {
        std::cout << "[BVH::build] Build complete! Total nodes: " << node_count << std::endl;
        if (root) {
            std::cout << "[BVH::build] Root node covers " 
                      << indices.size() << " spheres" << std::endl;
        } else {
            std::cout << "[BVH::build] ERROR: Root node is null!" << std::endl;
        }
    }
}

bool BVH::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec, 
             const std::vector<Sphere>& scene_spheres) const {
    // Debug only occasionally to avoid spam
    static int hit_count = 0;
    hit_count++;
    
    if (root == nullptr) {
        if (hit_count % 1000 == 0) {
            std::cout << "[BVH::hit] WARNING: Root is null! hit_count=" << hit_count << std::endl;
        }
        return false;
    }
    return root->hit(ray, t_min, t_max, rec, scene_spheres);
}