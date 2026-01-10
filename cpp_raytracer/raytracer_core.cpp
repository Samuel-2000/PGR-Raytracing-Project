#include "raytracer_core.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stack>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

// ================================================
// SIMPLIFIED BVH NODE FOR TRAVERSAL
// ================================================
struct TraversalNode {
    int node_index;
    float tmin;
    
    TraversalNode() : node_index(0), tmin(0.0f) {}  // Default constructor
    TraversalNode(int idx, float t) : node_index(idx), tmin(t) {}
};

// ================================================
// BVH BUILDER (Array-based, cache friendly)
// ================================================
class BVHBuilder {
private:
    struct BuildNode {
        AABB bbox;
        int start;
        int end;
        int parent;
        int depth;
        
        BuildNode() : start(0), end(0), parent(-1), depth(0) {}
        BuildNode(int s, int e, int p, int d) : start(s), end(e), parent(p), depth(d) {}
    };
    
    Sphere* spheres;
    int* indices;
    int n_spheres;
    BVHNodeFlat* flat_nodes;
    int node_count;
    int max_depth;
    
    FORCEINLINE bool box_compare(int idx_a, int idx_b, int axis) {
        if (axis == 0) return spheres[idx_a].center.x < spheres[idx_b].center.x;
        if (axis == 1) return spheres[idx_a].center.y < spheres[idx_b].center.y;
        return spheres[idx_a].center.z < spheres[idx_b].center.z;
    }
    
public:
    BVHBuilder(Sphere* spheres_ptr, int* indices_ptr, int n) 
        : spheres(spheres_ptr), indices(indices_ptr), n_spheres(n), 
          flat_nodes(nullptr), node_count(0), max_depth(0) {}
    
    int build(BVHNodeFlat* nodes, int max_nodes) {
        flat_nodes = nodes;
        node_count = 0;
        max_depth = 0;
        
        if (n_spheres == 0) return 0;
        
        // Build tree iteratively using stack (no recursion)
        std::stack<BuildNode> node_stack;
        node_stack.push(BuildNode(0, n_spheres, -1, 0));
        
        int node_index = 0;
        
        while (!node_stack.empty()) {
            BuildNode current = node_stack.top();
            node_stack.pop();
            
            int current_node_idx = node_index++;
            int span = current.end - current.start;
            
            // Calculate bounding box for this node
            AABB node_bbox;
            if (span > 0) {
                node_bbox = spheres[indices[current.start]].bbox;
                for (int i = current.start + 1; i < current.end; ++i) {
                    node_bbox = AABB::surrounding(node_bbox, spheres[indices[i]].bbox);
                }
            }
            
            if (span <= 4) {  // Create leaf
                flat_nodes[current_node_idx].bbox = node_bbox;
                flat_nodes[current_node_idx].first_primitive = current.start;
                flat_nodes[current_node_idx].primitive_count = span;
                
                if (current.depth > max_depth) max_depth = current.depth;
                continue;
            }
            
            // Find split axis (longest extent)
            Vector3 extent = node_bbox.max - node_bbox.min;
            int axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > extent.y && extent.z > extent.x) axis = 2;
            
            // Sort primitives
            std::sort(indices + current.start, indices + current.end,
                [this, axis](int a, int b) { return box_compare(a, b, axis); });
            
            int mid = current.start + span / 2;
            
            // Create internal node
            flat_nodes[current_node_idx].bbox = node_bbox;
            flat_nodes[current_node_idx].primitive_count = 0;  // Mark as internal
            
            // Push children (right first, then left for stack order)
            node_stack.push(BuildNode(mid, current.end, current_node_idx, current.depth + 1));
            node_stack.push(BuildNode(current.start, mid, current_node_idx, current.depth + 1));
            
            // Store child indices (will be updated after building)
            flat_nodes[current_node_idx].left_child = -1;
            flat_nodes[current_node_idx].right_child = -1;
        }
        
        // Second pass to assign child indices
        std::queue<int> node_queue;
        node_queue.push(0);
        int processed = 0;
        
        while (!node_queue.empty()) {
            int node_idx = node_queue.front();
            node_queue.pop();
            
            BVHNodeFlat& node = flat_nodes[node_idx];
            
            if (!node.is_leaf()) {
                node.left_child = ++processed;
                node.right_child = ++processed;
                node_queue.push(node.left_child);
                node_queue.push(node.right_child);
            }
        }
        
        node_count = node_index;
        return node_count;
    }
    
    int get_node_count() const { return node_count; }
    int get_max_depth() const { return max_depth; }
};

// ================================================
// SCENE INTERSECTOR
// ================================================
class SceneIntersector {
private:
    Sphere* spheres;
    int sphere_count;
    BVHNodeFlat* bvh_nodes;
    int* indices;
    int node_count;
    
public:
    SceneIntersector() : spheres(nullptr), sphere_count(0), 
                        bvh_nodes(nullptr), indices(nullptr), node_count(0) {}
    
    ~SceneIntersector() {
        delete[] bvh_nodes;
        delete[] indices;
    }
    
    void build_bvh(Sphere* scene_spheres, int count) {
        spheres = scene_spheres;
        sphere_count = count;
        
        if (count == 0) return;
        
        // Allocate indices array
        delete[] indices;
        indices = new int[count];
        for (int i = 0; i < count; ++i) indices[i] = i;
        
        // Allocate BVH nodes (max 2n - 1)
        delete[] bvh_nodes;
        int max_nodes = 2 * count - 1;
        bvh_nodes = new BVHNodeFlat[max_nodes];
        
        // Build BVH
        BVHBuilder builder(spheres, indices, count);
        node_count = builder.build(bvh_nodes, max_nodes);
        
        std::cout << "BVH built with " << node_count << " nodes, max depth: " 
                  << builder.get_max_depth() << std::endl;
    }
    
    FORCEINLINE bool intersect(const Ray& ray, float tmin, float tmax,
                              float& hit_t, Vector3& hit_normal, 
                              Material& hit_mat, int& hit_id) const {
        if (sphere_count == 0) return false;
        
        if (bvh_nodes && node_count > 0) {
            // Use BVH traversal
            TraversalNode stack[64];  // Stack on local memory (no heap allocation)
            int stack_ptr = 0;
            stack[stack_ptr++] = TraversalNode(0, tmin);
            
            bool hit = false;
            float closest_t = tmax;
            Vector3 closest_normal;
            Material closest_mat;
            int closest_id;
            
            while (stack_ptr > 0) {
                TraversalNode tnode = stack[--stack_ptr];
                
                // Early termination
                if (tnode.tmin >= closest_t) continue;
                
                const BVHNodeFlat& node = bvh_nodes[tnode.node_index];
                
                // Skip if no intersection
                if (!node.bbox.intersect(ray, tnode.tmin, closest_t)) continue;
                
                if (node.is_leaf()) {
                    // Test all primitives in leaf
                    for (int i = 0; i < node.primitive_count; ++i) {
                        int idx = indices[node.first_primitive + i];
                        const Sphere& sphere = spheres[idx];
                        
                        float t;
                        Vector3 normal;
                        Material mat;
                        int id;
                        
                        if (sphere.intersect(ray, tnode.tmin, closest_t, t, normal, mat, id)) {
                            closest_t = t;
                            closest_normal = normal;
                            closest_mat = mat;
                            closest_id = id;
                            hit = true;
                        }
                    }
                } else {
                    // Push children
                    stack[stack_ptr++] = TraversalNode(node.left_child, tnode.tmin);
                    stack[stack_ptr++] = TraversalNode(node.right_child, tnode.tmin);
                }
            }
            
            if (hit) {
                hit_t = closest_t;
                hit_normal = closest_normal;
                hit_mat = closest_mat;
                hit_id = closest_id;
                return true;
            }
        } else {
            // Brute force fallback
            float closest_t = tmax;
            for (int i = 0; i < sphere_count; ++i) {
                float t;
                Vector3 normal;
                Material mat;
                int id;
                
                if (spheres[i].intersect(ray, tmin, closest_t, t, normal, mat, id)) {
                    closest_t = t;
                    hit_t = t;
                    hit_normal = normal;
                    hit_mat = mat;
                    hit_id = id;
                    return true;
                }
            }
        }
        
        return false;
    }
};

// ================================================
// PATH TRACER (Iterative, no recursion)
// ================================================
class PathTracer {
private:
    SceneIntersector scene_intersector;
    Vector3 background_color;
    
public:
    PathTracer() : background_color(0.1f, 0.1f, 0.1f) {}
    
    void set_scene(Sphere* spheres, int count) {
        scene_intersector.build_bvh(spheres, count);
    }
    
    FORCEINLINE Vector3 trace_ray(const Ray& ray, int max_depth, PCG32& rng) {
        Vector3 color(0, 0, 0);
        Vector3 throughput(1, 1, 1);
        Ray current_ray = ray;
        int depth = 0;
        
        while (depth < max_depth) {
            depth++;
            
            float t;
            Vector3 normal;
            Material mat;
            int id;
            
            // Intersection test
            if (!scene_intersector.intersect(current_ray, 0.001f, 1e10f, 
                                           t, normal, mat, id)) {
                // Ray missed - add background
                color = color + throughput * background_color;
                break;
            }
            
            // Add emitted light
            color = color + throughput * mat.emission;
            
            // Russian Roulette termination
            if (depth > 3) {
                float max_component = (throughput.x > throughput.y) ? 
                    (throughput.x > throughput.z ? throughput.x : throughput.z) :
                    (throughput.y > throughput.z ? throughput.y : throughput.z);
                
                float continue_probability = (max_component > 0.95f) ? 0.95f : max_component;
                if (continue_probability < 0.1f) continue_probability = 0.1f;
                
                if (rng.random_float() >= continue_probability) {
                    break;
                }
                throughput = throughput / continue_probability;
            }
            
            Vector3 hit_point = current_ray.at(t);
            
            // Material scattering
            if (mat.metallic > 0.0f) {
                // Metallic reflection
                Vector3 reflected = FastMath::reflect(current_ray.direction.normalize(), normal);
                Vector3 random_scatter = FastMath::random_in_unit_sphere(rng) * mat.roughness;
                Vector3 new_direction = (reflected + random_scatter).normalize();
                current_ray = Ray(hit_point, new_direction);
                throughput = throughput * mat.albedo;
            } else {
                // Diffuse reflection
                Vector3 random_dir = FastMath::random_in_hemisphere(normal, rng);
                Vector3 new_direction = (normal + random_dir).normalize();
                current_ray = Ray(hit_point, new_direction);
                throughput = throughput * mat.albedo;
            }
        }
        
        return color;
    }
    
    // Parallel rendering with OpenMP static scheduling
    void render(float* image_data, int width, int height, 
                int samples_per_pixel, int max_depth, const Camera& camera) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Precompute 1/width and 1/height
        float inv_width = 1.0f / width;
        float inv_height = 1.0f / height;
        
        int total_pixels = width * height;
        
        #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        std::cout << "Rendering with " << num_threads << " threads" << std::endl;
        #pragma omp parallel
        #endif
        {
            #ifdef _OPENMP
            int thread_id = omp_get_thread_num();
            #else
            int thread_id = 0;
            #endif
            
            // Each thread gets its own RNG with different seed
            PCG32 rng(thread_id + 1);
            
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int pixel_idx = 0; pixel_idx < total_pixels; ++pixel_idx) {
                int j = pixel_idx / width;
                int i = pixel_idx % width;
                
                Vector3 pixel_color(0, 0, 0);
                
                for (int s = 0; s < samples_per_pixel; ++s) {
                    // Jittered sampling
                    float u = (i + rng.random_float()) * inv_width;
                    float v = (j + rng.random_float()) * inv_height;
                    
                    Ray ray = camera.get_ray(u, v);
                    pixel_color = pixel_color + trace_ray(ray, max_depth, rng);
                }
                
                pixel_color = pixel_color * (1.0f / samples_per_pixel);
                
                // Fast gamma correction (sqrt)
                pixel_color = Vector3(sqrtf(pixel_color.x), 
                                     sqrtf(pixel_color.y), 
                                     sqrtf(pixel_color.z));
                
                // Clamp and store
                int idx = (j * width + i) * 3;
                image_data[idx] = pixel_color.x < 0.0f ? 0.0f : (pixel_color.x > 1.0f ? 1.0f : pixel_color.x);
                image_data[idx + 1] = pixel_color.y < 0.0f ? 0.0f : (pixel_color.y > 1.0f ? 1.0f : pixel_color.y);
                image_data[idx + 2] = pixel_color.z < 0.0f ? 0.0f : (pixel_color.z > 1.0f ? 1.0f : pixel_color.z);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Render time: " << duration.count() << "ms" << std::endl;
    }
};

// ================================================
// PYTHON BINDING INTERFACE
// ================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class RayTracerWrapper {
private:
    Sphere* spheres;
    int sphere_count;
    PathTracer tracer;
    Camera camera;
    
public:
    RayTracerWrapper() : spheres(nullptr), sphere_count(0) {}
    
    ~RayTracerWrapper() {
        delete[] spheres;
    }
    
    void set_spheres(py::list sphere_list) {
        sphere_count = (int)sphere_list.size();
        delete[] spheres;
        spheres = new Sphere[sphere_count];
        
        for (int i = 0; i < sphere_count; ++i) {
            py::tuple sphere_data = sphere_list[i].cast<py::tuple>();
            
            // Unpack sphere data: (center, radius, albedo, metallic, roughness, emission)
            py::tuple center = sphere_data[0].cast<py::tuple>();
            float radius = sphere_data[1].cast<float>();
            py::tuple albedo = sphere_data[2].cast<py::tuple>();
            float metallic = sphere_data[3].cast<float>();
            float roughness = sphere_data[4].cast<float>();
            py::tuple emission = sphere_data[5].cast<py::tuple>();
            
            Material mat;
            mat.albedo = Vector3(albedo[0].cast<float>(), 
                                albedo[1].cast<float>(), 
                                albedo[2].cast<float>());
            mat.metallic = metallic;
            mat.roughness = roughness;
            mat.emission = Vector3(emission[0].cast<float>(),
                                  emission[1].cast<float>(),
                                  emission[2].cast<float>());
            
            spheres[i] = Sphere(
                Vector3(center[0].cast<float>(),
                       center[1].cast<float>(),
                       center[2].cast<float>()),
                radius,
                mat,
                i
            );
        }
        
        tracer.set_scene(spheres, sphere_count);
    }
    
    void set_camera(py::tuple pos, py::tuple target, float fov, float aspect) {
        camera.position = Vector3(pos[0].cast<float>(), 
                                 pos[1].cast<float>(), 
                                 pos[2].cast<float>());
        camera.fov = fov;
        camera.aspect_ratio = aspect;
        camera.update_basis();
    }
    
    py::array_t<float> render(int width, int height, int samples, int max_depth) {
        // Allocate output array
        auto result = py::array_t<float>({height, width, 3});
        auto buf = result.request();
        float* image_data = static_cast<float*>(buf.ptr);
        
        tracer.render(image_data, width, height, samples, max_depth, camera);
        
        return result;
    }
};

PYBIND11_MODULE(raytracer_cpp, m) {
    m.doc() = "High-performance ray tracer with AVX2 and OpenMP";
    
    py::class_<RayTracerWrapper>(m, "RayTracer")
        .def(py::init<>())
        .def("set_spheres", &RayTracerWrapper::set_spheres)
        .def("set_camera", &RayTracerWrapper::set_camera)
        .def("render", &RayTracerWrapper::render);
}