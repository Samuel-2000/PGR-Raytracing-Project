#include "raytracer_core.h"
#include "bvh.h"
#include <algorithm>
#include <iostream>
#include <chrono>

bool Sphere::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    Vector3 oc = ray.origin - center;
    double a = ray.direction.dot(ray.direction);
    double half_b = oc.dot(ray.direction);
    double c = oc.dot(oc) - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return false;
    }
    
    double sqrtd = std::sqrt(discriminant);
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.point = ray.at(rec.t);
    Vector3 outward_normal = (rec.point - center) / radius;
    rec.set_face_normal(ray, outward_normal);
    rec.material = material;
    rec.object_id = object_id;
    return true;
}

Scene::Scene() : background_color(0.1, 0.1, 0.1), bvh(nullptr), use_bvh(true) {}

Scene::~Scene() {
    delete bvh;
}

void Scene::add_sphere(const Sphere& sphere) {
    spheres.push_back(sphere);
}

void Scene::build_bvh() {
    if (bvh != nullptr) {
        delete bvh;
    }
    bvh = new BVH();
    bvh->build(spheres);
}

bool Scene::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (use_bvh && bvh != nullptr) {
        return bvh->hit(ray, t_min, t_max, rec);
    }
    
    // Fallback to brute force
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (const auto& sphere : spheres) {
        if (sphere.hit(ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

RayTracer::RayTracer() : gen(std::random_device{}()), dis(0.0, 1.0) {}

RayTracer::~RayTracer() {}

void RayTracer::set_scene(const Scene& new_scene) {
    scene = new_scene;
}

Vector3 RayTracer::random_in_unit_sphere() {
    Vector3 p;
    do {
        p = Vector3(dis(gen), dis(gen), dis(gen)) * 2.0 - Vector3(1, 1, 1);
    } while (p.length_squared() >= 1.0);
    return p;
}

Vector3 RayTracer::random_in_hemisphere(const Vector3& normal) {
    Vector3 in_unit_sphere = random_in_unit_sphere();
    if (in_unit_sphere.dot(normal) > 0.0) {
        return in_unit_sphere;
    }
    else {
        return in_unit_sphere * -1.0;
    }
}

Vector3 RayTracer::reflect(const Vector3& v, const Vector3& n) {
    return v - n * (2.0 * v.dot(n));
}

bool RayTracer::refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted) {
    Vector3 uv = v.normalize();
    double dt = uv.dot(n);
    double discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = (uv - n * dt) * ni_over_nt - n * std::sqrt(discriminant);
        return true;
    }
    return false;
}

double RayTracer::schlick(double cosine, double ref_idx) {
    double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
}

Vector3 RayTracer::trace_ray(const Ray& ray, int depth, int max_depth) {
    if (depth <= 0) {
        return Vector3(0, 0, 0);
    }
    
    HitRecord rec;
    if (scene.hit(ray, 0.001, 1e10, rec)) {
        Vector3 emitted = rec.material.emission;
        
        // Russian Roulette path termination
        double continue_probability = 0.8;
        if (depth < 3 || dis(gen) < continue_probability) {
            if (dis(gen) < rec.material.metallic) {
                // Metallic reflection
                Vector3 reflected = reflect(ray.direction.normalize(), rec.normal);
                Vector3 random_scatter = random_in_unit_sphere() * rec.material.roughness;
                Ray scattered(rec.point, reflected + random_scatter);
                Vector3 traced_color = trace_ray(scattered, depth - 1, max_depth);
                return emitted + (traced_color * rec.material.albedo);
            }
            else {
                // Diffuse reflection
                Vector3 target = rec.point + rec.normal + random_in_hemisphere(rec.normal);
                Ray scattered(rec.point, target - rec.point);
                Vector3 traced_color = trace_ray(scattered, depth - 1, max_depth);
                return emitted + (traced_color * rec.material.albedo);
            }
        }
        return emitted;
    }
    
    return scene.background_color;
}

std::vector<double> RayTracer::render(int width, int height, int samples_per_pixel, int max_depth) {
    std::vector<double> image_data(width * height * 3);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simple parallelization without collapse for better compatibility
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int j = 0; j < height; ++j) {
        // Create local random number generator for each thread
        std::mt19937 local_gen(gen() + j);
        std::uniform_real_distribution<double> local_dis(0.0, 1.0);
        
        for (int i = 0; i < width; ++i) {
            Vector3 pixel_color(0, 0, 0);
            
            for (int s = 0; s < samples_per_pixel; ++s) {
                double u = (double(i) + local_dis(local_gen)) / double(width);
                double v = (double(j) + local_dis(local_gen)) / double(height);
                
                // Flip the v coordinate to fix upside-down image
                // Original: v goes from 0 (top) to 1 (bottom)
                // Fixed: v goes from 0 (bottom) to 1 (top)
                double flipped_v = 1.0 - v;
                
                Ray ray(Vector3(0, 0, 0), Vector3(u - 0.5, flipped_v - 0.5, -1.0));
                pixel_color = pixel_color + trace_ray(ray, max_depth, max_depth);
            }
            
            pixel_color = pixel_color * (1.0 / double(samples_per_pixel));
            
            // Gamma correction
            pixel_color = Vector3(
                std::sqrt(pixel_color.x),
                std::sqrt(pixel_color.y),
                std::sqrt(pixel_color.z)
            );
            
            int idx = (j * width + i) * 3;
            image_data[idx] = std::min(1.0, std::max(0.0, pixel_color.x));
            image_data[idx + 1] = std::min(1.0, std::max(0.0, pixel_color.y));
            image_data[idx + 2] = std::min(1.0, std::max(0.0, pixel_color.z));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Render time: " << duration.count() << "ms" << std::endl;
    
    return image_data;
}