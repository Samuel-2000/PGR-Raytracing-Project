#pragma once
#include <vector>
#include <cmath>
#include <random>

struct Vector3 {
    double x, y, z;
    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    
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
    double length() const { 
        return std::sqrt(x*x + y*y + z*z); 
    }
    double length_squared() const { 
        return x*x + y*y + z*z; 
    }
    Vector3 normalize() const { 
        double len = length(); 
        return (len > 0) ? Vector3(x/len, y/len, z/len) : *this; 
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
};

class RayTracer {
private:
    Scene scene;
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
};