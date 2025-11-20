#pragma once
#include <vector>
#include <cmath>
#include <random>

struct Vector3 {
    double x, y, z;
    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
    // Add array access operators
    double operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    double& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    Vector3 operator+(const Vector3& other) const { 
        return Vector3(x + other.x, y + other.y, z + other.z); 
    }
    Vector3 operator-(const Vector3& other) const { 
        return Vector3(x - other.x, y - other.y, z - other.z); 
    }
    Vector3 operator*(double scalar) const { 
        return Vector3(x * scalar, y * scalar, z * scalar); 
    }
    Vector3 operator*(const Vector3& other) const { 
        return Vector3(x * other.x, y * other.y, z * other.z); 
    }
    Vector3 operator/(double scalar) const { 
        return Vector3(x / scalar, y / scalar, z / scalar); 
    }
    Vector3 operator-() const { 
        return Vector3(-x, -y, -z); 
    }
    
    Vector3& operator+=(const Vector3& other) { 
        x += other.x; y += other.y; z += other.z; return *this; 
    }
    Vector3& operator*=(double scalar) { 
        x *= scalar; y *= scalar; z *= scalar; return *this; 
    }

    double dot(const Vector3& other) const { 
        return x * other.x + y * other.y + z * other.z; 
    }
    Vector3 cross(const Vector3& other) const { 
        return Vector3(y * other.z - z * other.y, 
                      z * other.x - x * other.z, 
                      x * other.y - y * other.x);
    }
    double length_squared() const { 
        return x*x + y*y + z*z; 
    }
    
    double length() const { 
        return std::sqrt(length_squared()); 
    }
    
    Vector3 normalize() const { 
        double len = length(); 
        if (len > 0) {
            return Vector3(x/len, y/len, z/len);
        }
        return *this;
    }
};



struct Ray {
    Vector3 origin;
    Vector3 direction;
    Ray(const Vector3& orig, const Vector3& dir) : origin(orig), direction(dir.normalize()) {}
    Vector3 at(double t) const { 
        return origin + direction * t; 
    }
};

struct Material {
    Vector3 albedo;
    double metallic;
    double roughness;
    Vector3 emission;
    double ior;
    
    Material() : albedo(0.8, 0.8, 0.8), metallic(0.0), roughness(0.5), 
                emission(0,0,0), ior(1.5) {}
};

struct HitRecord {
    double t;
    Vector3 point;
    Vector3 normal;
    Material material;
    bool front_face;
    int object_id;
    
    HitRecord() : t(0), point(0,0,0), normal(0,0,0), material(), 
                 front_face(true), object_id(0) {}
    
    void set_face_normal(const Ray& ray, const Vector3& outward_normal) {
        front_face = ray.direction.dot(outward_normal) < 0;
        normal = front_face ? outward_normal : outward_normal * -1.0;
    }
};

struct Sphere {
    Vector3 center;
    double radius;
    Material material;
    int object_id;
    std::string name;
    
    Sphere() : center(0,0,0), radius(1.0), material(), object_id(0), name("") {}
    
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
};

class Camera {
public:
    Vector3 position;
    Vector3 target;
    Vector3 up;
    double fov;
    double aspect_ratio;
    
    Camera() : position(0, 2, 3), target(0, 0, -3), up(0, 1, 0), fov(45.0), aspect_ratio(1.333) {}
    
    Ray get_ray(double u, double v) const {
        // Convert from [0,1] to [-1,1] and account for aspect ratio
        double ndc_x = (u - 0.5) * 2.0;
        double ndc_y = (0.5 - v) * 2.0;  // Flip Y
        
        double tan_fov = std::tan(fov * 3.14159 / 360.0);
        double view_x = ndc_x * aspect_ratio * tan_fov;
        double view_y = ndc_y * tan_fov;
        
        Vector3 direction(view_x, view_y, -1.0);
        direction = direction.normalize();
        
        return Ray(position, direction);
    }
    
    void move(const Vector3& delta) {
        position = position + delta;
    }
    
    void rotate(double dx, double dy) {
        // Simple rotation around target
        Vector3 forward = (target - position).normalize();
        Vector3 right = forward.cross(up).normalize();
        
        // Rotate position around target
        double distance = (target - position).length();
        Vector3 offset = position - target;
        
        // Apply rotation (simplified)
        position = target + offset;
    }
};


class BVH;

class Scene {
public:
    std::vector<Sphere> spheres;
    Vector3 background_color;
    BVH* bvh;
    bool use_bvh;
    
    Scene();
    ~Scene();
    
    void add_sphere(const Sphere& sphere);
    void build_bvh();
    bool hit(const Ray& ray, double t_min, double t_max, HitRecord& rec) const;
    
    // New method for object selection
    int cast_ray_for_selection(const Ray& ray, double t_min, double t_max) const;
};

class RayTracer {
private:
    Scene scene;
    Camera camera;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
    Vector3 random_in_unit_sphere();
    Vector3 random_in_hemisphere(const Vector3& normal);
    Vector3 reflect(const Vector3& v, const Vector3& n);
    bool refract(const Vector3& v, const Vector3& n, double ni_over_nt, Vector3& refracted);
    double schlick(double cosine, double ref_idx);
    
public:
    RayTracer();
    ~RayTracer();
    void set_scene(const Scene& new_scene);
    Vector3 trace_ray(const Ray& ray, int depth, int max_depth);
    std::vector<double> render(int width, int height, int samples_per_pixel, int max_depth);
    
    // New methods for interaction
    Camera& get_camera() { return camera; }
    int select_object(double x, double y, int width, int height);
    void move_camera(const Vector3& delta);
};