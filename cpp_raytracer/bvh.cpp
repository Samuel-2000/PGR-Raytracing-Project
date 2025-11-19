#include "bvh.h"
#include <algorithm>

bool AABB::hit(const Ray& ray, double tmin, double tmax) const {
    for (int a = 0; a < 3; a++) {
        double invD = 1.0 / ray.direction.x;
        double t0 = (min.x - ray.origin.x) * invD;
        double t1 = (max.x - ray.origin.x) * invD;
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

bool BVHNode::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (!box.hit(ray, t_min, t_max)) {
        return false;
    }
    
    if (is_leaf) {
        return sphere.hit(ray, t_min, t_max, rec);
    }
    
    HitRecord left_rec, right_rec;
    bool hit_left = (left != nullptr) && left->hit(ray, t_min, t_max, left_rec);
    bool hit_right = (right != nullptr) && right->hit(ray, t_min, t_max, right_rec);
    
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
    
    if (axis == 0) {
        return box_a.min.x < box_b.min.x;
    }
    else if (axis == 1) {
        return box_a.min.y < box_b.min.y;
    }
    else {
        return box_a.min.z < box_b.min.z;
    }
}

BVHNode* BVH::build_tree(std::vector<Sphere>& spheres, size_t start, size_t end, int depth) {
    if (start >= end) {
        return nullptr; // No spheres in this range
    }
    
    BVHNode* node = new BVHNode();
    size_t span = end - start;
    
    // For small numbers of spheres, create a leaf node containing all of them
    if (span <= 2) {
        // Calculate bounding box for all spheres in this leaf
        AABB box = sphere_bounding_box(spheres[start]);
        for (size_t i = start + 1; i < end; i++) {
            box = AABB::surrounding_box(box, sphere_bounding_box(spheres[i]));
        }
        node->box = box;
        
        // Store all spheres in this leaf (you'll need to modify BVHNode structure)
        // For now, we'll handle the single sphere case
        if (span == 1) {
            node->sphere = spheres[start];
            node->is_leaf = true;
        } else {
            // For two spheres, create two child leaf nodes
            node->left = build_tree(spheres, start, start + 1, depth + 1);
            node->right = build_tree(spheres, start + 1, end, depth + 1);
            node->box = AABB::surrounding_box(node->left->box, node->right->box);
            node->is_leaf = false;
        }
        return node;
    }
    
    // Calculate total bounding box
    AABB total_box = sphere_bounding_box(spheres[start]);
    for (size_t i = start + 1; i < end; i++) {
        total_box = AABB::surrounding_box(total_box, sphere_bounding_box(spheres[i]));
    }
    node->box = total_box;
    
    // Choose split axis based on largest extent
    Vector3 extent = total_box.max - total_box.min;
    int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 : 
               (extent.y > extent.z) ? 1 : 2;
    
    // Sort only the portion we're working with
    auto comparator = [axis, this](const Sphere& a, const Sphere& b) {
        return this->box_compare(a, b, axis);
    };
    std::sort(spheres.begin() + start, spheres.begin() + end, comparator);
    
    // Split at midpoint
    size_t mid = start + span / 2;
    
    // Recursively build children
    node->left = build_tree(spheres, start, mid, depth + 1);
    node->right = build_tree(spheres, mid, end, depth + 1);
    node->is_leaf = false;
    
    // Handle case where one child might be null
    if (node->left == nullptr) {
        BVHNode* temp = node->right;
        delete node;
        return temp;
    }
    if (node->right == nullptr) {
        BVHNode* temp = node->left;
        delete node;
        return temp;
    }
    
    return node;
}

BVH::BVH() : root(nullptr) {}

BVH::~BVH() {
    delete root;
}

void BVH::build(const std::vector<Sphere>& spheres) {
    if (spheres.empty()) {
        return;
    }
    
    std::vector<Sphere> mutable_spheres = spheres;
    root = build_tree(mutable_spheres, 0, mutable_spheres.size());
}

bool BVH::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (root == nullptr) {
        return false;
    }
    return root->hit(ray, t_min, t_max, rec);
}