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
    BVHNode* node = new BVHNode();
    
    // Limit recursion depth to prevent stack overflow
    if (depth > 100) {
        // Create leaf node with all remaining spheres
        // For simplicity, just take the first sphere
        if (start < end) {
            node->sphere = spheres[start];
            node->box = sphere_bounding_box(spheres[start]);
            node->is_leaf = true;
        }
        return node;
    }
    
    size_t span = end - start;
    if (span == 1) {
        // Leaf node with single sphere
        node->sphere = spheres[start];
        node->box = sphere_bounding_box(spheres[start]);
        node->is_leaf = true;
        return node;
    }
    else if (span == 2) {
        // Create two leaf nodes
        node->left = build_tree(spheres, start, start + 1, depth + 1);
        node->right = build_tree(spheres, start + 1, end, depth + 1);
        node->box = AABB::surrounding_box(node->left->box, node->right->box);
        node->is_leaf = false;
        return node;
    }
    
    // Calculate bounding box for all spheres in this node
    AABB box = sphere_bounding_box(spheres[start]);
    for (size_t i = start + 1; i < end; i++) {
        box = AABB::surrounding_box(box, sphere_bounding_box(spheres[i]));
    }
    node->box = box;
    
    // Choose split axis based on the largest extent
    Vector3 extent = box.max - box.min;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent.y && extent.z > extent.x) axis = 2;
    
    // Sort spheres along the chosen axis
    auto comparator = [axis, this](const Sphere& a, const Sphere& b) {
        return this->box_compare(a, b, axis);
    };
    std::sort(spheres.begin() + start, spheres.begin() + end, comparator);
    
    size_t mid = start + span / 2;
    node->left = build_tree(spheres, start, mid, depth + 1);
    node->right = build_tree(spheres, mid, end, depth + 1);
    node->is_leaf = false;
    
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