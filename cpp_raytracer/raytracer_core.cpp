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

Scene::Scene() : background_color(0.1, 0.1, 0.1), bvh(nullptr), use_bvh(true), debug_mode(false) {}

Scene::~Scene() {
    delete bvh;
}

void Scene::add_sphere(const Sphere& sphere) {
    spheres.push_back(sphere);
    if (debug_mode) {
        std::cout << "[Scene] Added sphere: " << sphere.name 
                  << " id=" << sphere.object_id 
                  << " pos=(" << sphere.center.x << "," << sphere.center.y << "," << sphere.center.z 
                  << ") radius=" << sphere.radius << std::endl;
    }
}


void Scene::build_bvh() {
    if (debug_mode) {
        std::cout << "\n[Scene] Building BVH with " << spheres.size() << " spheres:" << std::endl;
        for (size_t i = 0; i < spheres.size(); i++) {
            const auto& s = spheres[i];
            std::cout << "  [" << i << "] " << s.name 
                      << " id=" << s.object_id
                      << " pos=(" << s.center.x << "," << s.center.y << "," << s.center.z << ")" << std::endl;
        }
    }
    
    if (bvh != nullptr) {
        delete bvh;
    }
    bvh = new BVH();
    bvh->build(spheres, debug_mode);
}

// In Scene::hit method
bool Scene::hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    if (use_bvh && bvh != nullptr) {
        bool hit = bvh->hit(ray, t_min, t_max, rec, spheres);
        if (debug_mode && hit) {
            std::cout << "[Scene::hit] BVH hit: object_id=" << rec.object_id 
                      << " t=" << rec.t << std::endl;
        }
        return hit;
    }
    
    // Fallback to brute force
    if (debug_mode) {
        std::cout << "[Scene::hit] Using brute force (BVH disabled or null)" << std::endl;
    }
    
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
    
    if (debug_mode && hit_anything) {
        std::cout << "[Scene::hit] Brute force hit: object_id=" << rec.object_id << std::endl;
    }
    
    return hit_anything;
}

int Scene::cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const {
    HitRecord rec;
    int selected_id = -1;
    double closest_t = t_max;

    if (debug_mode) {
        std::cout << "[Scene::cast_ray_for_selection] Ray from (" 
                  << ray.origin.x << "," << ray.origin.y << "," << ray.origin.z << ")" << std::endl;
    }

    for (const auto& sphere : spheres) {
        if (sphere.hit(ray, t_min, closest_t, rec)) {
            closest_t = rec.t;
            selected_id = sphere.object_id;
            if (debug_mode) {
                std::cout << "  Hit sphere: " << sphere.name 
                          << " id=" << sphere.object_id 
                          << " at t=" << rec.t << std::endl;
            }
        }
    }
    
    if (debug_mode) {
        std::cout << "[Scene::cast_ray_for_selection] Selected id: " << selected_id << std::endl;
    }
    
    return selected_id;
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

int RayTracer::select_object(double x, double y, int width, int height) {
    // Convert screen coordinates to ray
    Ray ray = camera.get_ray(x, y);
    return scene.cast_ray_for_selection(ray, 0.001, 1000.0);
}

void RayTracer::move_camera(const Vector3& delta) {
    camera.move(delta);
}

std::vector<double> RayTracer::render(int width, int height, int samples_per_pixel, int max_depth) {
    std::vector<double> image_data(width * height * 3);
    camera.aspect_ratio = static_cast<double>(width) / height;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int j = 0; j < height; ++j) {
        std::mt19937 local_gen(gen() + j);
        std::uniform_real_distribution<double> local_dis(0.0, 1.0);
        
        for (int i = 0; i < width; ++i) {
            Vector3 pixel_color(0, 0, 0);
            
            for (int s = 0; s < samples_per_pixel; ++s) {
                double u = (double(i) + local_dis(local_gen)) / double(width);
                double v = (double(j) + local_dis(local_gen)) / double(height);
                
                Ray ray = camera.get_ray(u, v);
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