#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "raytracer_core.h"

namespace py = pybind11;

PYBIND11_MODULE(raytracer_cpp, m) {
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z)
        .def("__repr__", [](const Vector3& v) {
            return "Vector3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });
    
    py::class_<Ray>(m, "Ray")
        .def(py::init<const Vector3&, const Vector3&>())
        .def_readwrite("origin", &Ray::origin)
        .def_readwrite("direction", &Ray::direction);
    
    py::class_<Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("albedo", &Material::albedo)
        .def_readwrite("metallic", &Material::metallic)
        .def_readwrite("roughness", &Material::roughness)
        .def_readwrite("emission", &Material::emission)
        .def_readwrite("ior", &Material::ior);
    
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("material", &Sphere::material)
        .def_readwrite("object_id", &Sphere::object_id)
        .def_readwrite("name", &Sphere::name);
    
    py::class_<Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("position", &Camera::position)
        .def_readwrite("target", &Camera::target)
        .def_readwrite("up", &Camera::up)
        .def_readwrite("fov", &Camera::fov)
        .def("get_ray", &Camera::get_ray)
        .def("move", &Camera::move);
    
    py::class_<Scene>(m, "Scene")
        .def(py::init<>())
        .def_readwrite("spheres", &Scene::spheres)
        .def_readwrite("background_color", &Scene::background_color)
        .def_readwrite("use_bvh", &Scene::use_bvh)
        .def("add_sphere", &Scene::add_sphere)
        .def("build_bvh", &Scene::build_bvh)
        .def("cast_ray_for_selection", &Scene::cast_ray_for_selection);
    
    py::class_<RayTracer>(m, "RayTracer")
        .def(py::init<>())
        .def("set_scene", &RayTracer::set_scene)
        .def("render", &RayTracer::render)
        .def("get_camera", &RayTracer::get_camera, py::return_value_policy::reference)
        .def("select_object", &RayTracer::select_object)
        .def("move_camera", &RayTracer::move_camera);
}