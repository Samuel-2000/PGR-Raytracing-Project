# FILE: materials.py
"""
Advanced PBR Material System with Physically Based Rendering
"""
import numpy as np
from typing import Tuple, Optional
import math

class Vector3:
    """Optimized 3D vector for material calculations"""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
    
    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def length_squared(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def normalize(self):
        length = self.length()
        if length > 1e-8:
            return self * (1.0 / length)
        return Vector3(0, 0, 1)
    
    def reflect(self, n):
        return self - n * (2 * self.dot(n))
    
    def refract(self, n, eta_ratio):
        cos_theta = min(-self.dot(n), 1.0)
        r_out_perp = (self + n * cos_theta) * eta_ratio
        r_out_parallel = n * -math.sqrt(abs(1.0 - r_out_perp.length_squared()))
        return r_out_perp + r_out_parallel
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr):
        return Vector3(arr[0], arr[1], arr[2])

class Material:
    """Base material class with common PBR properties"""
    
    def __init__(self, 
                 albedo: Vector3 = None,
                 metallic: float = 0.0,
                 roughness: float = 0.5,
                 emission: Vector3 = None,
                 ior: float = 1.5,
                 transmission: float = 0.0,
                 specular: float = 0.5):
        
        self.albedo = albedo if albedo else Vector3(0.8, 0.8, 0.8)
        self.metallic = max(0.0, min(1.0, metallic))
        self.roughness = max(0.0, min(1.0, roughness))
        self.emission = emission if emission else Vector3(0, 0, 0)
        self.ior = max(1.0, ior)
        self.transmission = max(0.0, min(1.0, transmission))
        self.specular = max(0.0, min(1.0, specular))
        
        # Precomputed values for performance
        self._alpha = self.roughness * self.roughness
        self._alpha2 = self._alpha * self._alpha
    
    def evaluate_brdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> Vector3:
        """Evaluate BRDF for given directions"""
        if self.metallic > 0.0:
            return self._evaluate_metallic_brdf(wo, wi, normal)
        else:
            return self._evaluate_dielectric_brdf(wo, wi, normal)
    
    def sample_direction(self, wo: Vector3, normal: Vector3, r1: float, r2: float) -> Tuple[Vector3, float]:
        """Sample a new direction based on material properties"""
        if self.transmission > 0.0 and r1 < self.transmission:
            # Transmission (refraction)
            return self._sample_transmission(wo, normal, r2)
        elif self.metallic > 0.0 or self.specular > 0.0:
            # Metallic/specular reflection
            return self._sample_specular(wo, normal, r1, r2)
        else:
            # Diffuse reflection
            return self._sample_diffuse(normal, r1, r2)
    
    def pdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> float:
        """Probability density function for sampled direction"""
        if self.transmission > 0.0:
            # Mix between diffuse and transmission
            diffuse_pdf = self._diffuse_pdf(wi, normal)
            transmission_pdf = self._transmission_pdf(wo, wi, normal)
            return (1 - self.transmission) * diffuse_pdf + self.transmission * transmission_pdf
        elif self.metallic > 0.0:
            return self._specular_pdf(wo, wi, normal)
        else:
            return self._diffuse_pdf(wi, normal)
    
    def _evaluate_metallic_brdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> Vector3:
        """Evaluate metallic BRDF using Cook-Torrance model"""
        h = (wi + wo).normalize()
        ndotv = max(0.0, normal.dot(wo))
        ndotl = max(0.0, normal.dot(wi))
        ndoth = max(0.0, normal.dot(h))
        vdoth = max(0.0, wo.dot(h))
        
        if ndotl <= 0.0 or ndotv <= 0.0:
            return Vector3(0, 0, 0)
        
        # Fresnel term (Schlick's approximation)
        f0 = self.albedo * self.metallic + Vector3(0.04, 0.04, 0.04) * (1 - self.metallic)
        fresnel = f0 + (Vector3(1, 1, 1) - f0) * ((1 - vdoth) ** 5)
        
        # Normal distribution function (GGX/Trowbridge-Reitz)
        ndoth2 = ndoth * ndoth
        denom = ndoth2 * (self._alpha2 - 1) + 1
        ndf = self._alpha2 / (math.pi * denom * denom)
        
        # Geometry term (Smith with Schlick-GGX)
        k = (self.roughness + 1) * (self.roughness + 1) / 8
        geo_nv = ndotv / (ndotv * (1 - k) + k)
        geo_nl = ndotl / (ndotl * (1 - k) + k)
        geo = geo_nv * geo_nl
        
        # Cook-Torrance BRDF
        denominator = 4 * ndotv * ndotl
        if denominator <= 1e-8:
            return Vector3(0, 0, 0)
        
        specular = (fresnel * ndf * geo) / denominator
        return specular
    
    def _evaluate_dielectric_brdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> Vector3:
        """Evaluate dielectric BRDF"""
        h = (wi + wo).normalize()
        ndotv = max(0.0, normal.dot(wo))
        ndotl = max(0.0, normal.dot(wi))
        ndoth = max(0.0, normal.dot(h))
        vdoth = max(0.0, wo.dot(h))
        
        if ndotl <= 0.0 or ndotv <= 0.0:
            return Vector3(0, 0, 0)
        
        # Fresnel term for dielectric
        f0 = Vector3(0.04, 0.04, 0.04)
        fresnel = f0 + (Vector3(1, 1, 1) - f0) * ((1 - vdoth) ** 5)
        
        # Normal distribution function
        ndoth2 = ndoth * ndoth
        denom = ndoth2 * (self._alpha2 - 1) + 1
        ndf = self._alpha2 / (math.pi * denom * denom)
        
        # Geometry term
        k = (self.roughness + 1) * (self.roughness + 1) / 8
        geo_nv = ndotv / (ndotv * (1 - k) + k)
        geo_nl = ndotl / (ndotl * (1 - k) + k)
        geo = geo_nv * geo_nl
        
        # Specular component
        denominator = 4 * ndotv * ndotl
        if denominator <= 1e-8:
            specular = Vector3(0, 0, 0)
        else:
            specular = (fresnel * ndf * geo) / denominator
        
        # Diffuse component (Lambertian)
        diffuse = self.albedo * (1 - self.metallic) / math.pi
        
        # Mix based on specular
        return diffuse * (1 - self.specular) + specular * self.specular
    
    def _sample_diffuse(self, normal: Vector3, r1: float, r2: float) -> Tuple[Vector3, float]:
        """Cosine-weighted hemisphere sampling for diffuse"""
        # Cosine-weighted sampling in local coordinates
        z = math.sqrt(r1)
        r = math.sqrt(1 - r1)
        phi = 2 * math.pi * r2
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        
        # Transform to world coordinates
        local_dir = Vector3(x, y, z)
        world_dir = self._local_to_world(local_dir, normal)
        
        pdf = z / math.pi  # Cosine-weighted PDF
        
        return world_dir, pdf
    
    def _sample_specular(self, wo: Vector3, normal: Vector3, r1: float, r2: float) -> Tuple[Vector3, float]:
        """Importance sampling for specular reflection"""
        # Sample microfacet normal using GGX distribution
        phi = 2 * math.pi * r1
        cos_theta = math.sqrt((1 - r2) / (1 - r2 + r2 * self._alpha2))
        sin_theta = math.sqrt(1 - cos_theta * cos_theta)
        
        h_local = Vector3(
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta
        )
        
        # Transform to world coordinates
        h = self._local_to_world(h_local, normal)
        
        # Reflect wo around h
        wi = wo.reflect(h)
        
        # Calculate PDF
        ndoth = max(0.0, normal.dot(h))
        ndotv = max(0.0, normal.dot(wo))
        vdoth = max(0.0, wo.dot(h))
        
        if vdoth <= 0.0:
            return wi, 0.0
        
        # PDF for GGX distribution
        denom = ndoth * ndoth * (self._alpha2 - 1) + 1
        d_term = self._alpha2 / (math.pi * denom * denom)
        pdf_microfacet = d_term * ndoth
        
        # Convert to PDF for reflection direction
        pdf = pdf_microfacet / (4 * vdoth)
        
        return wi, pdf
    
    def _sample_transmission(self, wo: Vector3, normal: Vector3, r2: float) -> Tuple[Vector3, float]:
        """Sample transmission direction for refractive materials"""
        # Determine eta ratio
        cos_theta = -wo.dot(normal)
        entering = cos_theta > 0
        eta_i = 1.0 if entering else self.ior
        eta_t = self.ior if entering else 1.0
        eta_ratio = eta_i / eta_t
        
        # Sample refraction direction
        refracted = wo.refract(normal, eta_ratio)
        
        if refracted.length_squared() == 0:  # Total internal reflection
            reflected = wo.reflect(normal)
            return reflected, 1.0
        
        # Fresnel factor for PDF weighting
        f = self._fresnel_dielectric(cos_theta, eta_ratio)
        
        # Use r2 to choose between reflection and transmission
        if r2 < f:  # Reflect
            wi = wo.reflect(normal)
            pdf = f
        else:  # Transmit
            wi = refracted
            pdf = 1 - f
        
        return wi, pdf
    
    def _fresnel_dielectric(self, cos_theta: float, eta_ratio: float) -> float:
        """Calculate Fresnel factor for dielectric materials"""
        # Use Schlick's approximation
        r0 = ((1 - eta_ratio) / (1 + eta_ratio)) ** 2
        return r0 + (1 - r0) * ((1 - cos_theta) ** 5)
    
    def _local_to_world(self, local: Vector3, normal: Vector3) -> Vector3:
        """Transform from local coordinates to world coordinates"""
        # Create orthonormal basis
        if abs(normal.x) > abs(normal.y):
            tangent = Vector3(normal.z, 0, -normal.x).normalize()
        else:
            tangent = Vector3(0, -normal.z, normal.y).normalize()
        bitangent = normal.cross(tangent)
        
        # Transform
        world = (tangent * local.x + 
                bitangent * local.y + 
                normal * local.z)
        return world.normalize()
    
    def _diffuse_pdf(self, wi: Vector3, normal: Vector3) -> float:
        """PDF for diffuse sampling"""
        cosine = max(0.0, normal.dot(wi))
        return cosine / math.pi
    
    def _specular_pdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> float:
        """PDF for specular sampling"""
        h = (wi + wo).normalize()
        ndoth = max(0.0, normal.dot(h))
        vdoth = max(0.0, wo.dot(h))
        
        if vdoth <= 0.0:
            return 0.0
        
        denom = ndoth * ndoth * (self._alpha2 - 1) + 1
        d_term = self._alpha2 / (math.pi * denom * denom)
        return (d_term * ndoth) / (4 * vdoth)
    
    def _transmission_pdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> float:
        """PDF for transmission sampling"""
        cos_theta = -wo.dot(normal)
        eta_ratio = 1.0 / self.ior if cos_theta > 0 else self.ior
        f = self._fresnel_dielectric(cos_theta, eta_ratio)
        
        # Simple approximation - in practice this would be more complex
        return 1 - f

class MetalMaterial(Material):
    """Specialized metal material with enhanced properties"""
    
    def __init__(self, 
                 albedo: Vector3 = None,
                 roughness: float = 0.1,
                 f0: Vector3 = None,
                 anisotropic: float = 0.0):
        
        base_albedo = albedo if albedo else Vector3(0.9, 0.9, 0.9)
        super().__init__(albedo=base_albedo, metallic=1.0, roughness=roughness)
        
        self.f0 = f0 if f0 else base_albedo  # Base reflectivity
        self.anisotropic = max(0.0, min(1.0, anisotropic))
    
    def _evaluate_metallic_brdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> Vector3:
        """Enhanced metallic BRDF with anisotropic support"""
        h = (wi + wo).normalize()
        ndotv = max(0.0, normal.dot(wo))
        ndotl = max(0.0, normal.dot(wi))
        ndoth = max(0.0, normal.dot(h))
        vdoth = max(0.0, wo.dot(h))
        
        if ndotl <= 0.0 or ndotv <= 0.0:
            return Vector3(0, 0, 0)
        
        # Anisotropic roughness
        ax = max(0.001, self.roughness * (1 + self.anisotropic))
        ay = max(0.001, self.roughness * (1 - self.anisotropic))
        
        # Anisotropic NDF
        hdotx = h.dot(self._get_tangent(normal))
        hdoty = h.dot(self._get_bitangent(normal))
        denom = (hdotx / ax) ** 2 + (hdoty / ay) ** 2 + ndoth * ndoth
        ndf = 1.0 / (math.pi * ax * ay * denom * denom)
        
        # Fresnel with custom F0
        fresnel = self.f0 + (Vector3(1, 1, 1) - self.f0) * ((1 - vdoth) ** 5)
        
        # Geometry term
        k = (self.roughness + 1) * (self.roughness + 1) / 8
        geo_nv = ndotv / (ndotv * (1 - k) + k)
        geo_nl = ndotl / (ndotl * (1 - k) + k)
        geo = geo_nv * geo_nl
        
        denominator = 4 * ndotv * ndotl
        if denominator <= 1e-8:
            return Vector3(0, 0, 0)
        
        return (fresnel * ndf * geo) / denominator
    
    def _get_tangent(self, normal: Vector3) -> Vector3:
        """Get tangent vector for anisotropic calculations"""
        if abs(normal.x) > abs(normal.y):
            return Vector3(normal.z, 0, -normal.x).normalize()
        else:
            return Vector3(0, -normal.z, normal.y).normalize()
    
    def _get_bitangent(self, normal: Vector3) -> Vector3:
        """Get bitangent vector for anisotropic calculations"""
        tangent = self._get_tangent(normal)
        return normal.cross(tangent)

class GlassMaterial(Material):
    """Specialized glass material with proper refraction"""
    
    def __init__(self, 
                 ior: float = 1.5,
                 roughness: float = 0.0,
                 tint: Vector3 = None):
        
        super().__init__(
            albedo=Vector3(1, 1, 1),
            metallic=0.0,
            roughness=roughness,
            ior=ior,
            transmission=1.0,
            specular=1.0
        )
        self.tint = tint if tint else Vector3(1, 1, 1)
    
    def evaluate_brdf(self, wo: Vector3, wi: Vector3, normal: Vector3) -> Vector3:
        """Glass BRDF with proper transmission"""
        # For glass, we handle transmission separately in the integrator
        # This is a simplified version
        h = (wi + wo).normalize()
        ndotv = max(0.0, normal.dot(wo))
        ndotl = max(0.0, normal.dot(wi))
        
        if ndotl <= 0.0 or ndotv <= 0.0:
            return Vector3(0, 0, 0)
        
        # Perfect specular for now
        return Vector3(1, 1, 1) * self.specular

class EmissiveMaterial(Material):
    """Material that emits light"""
    
    def __init__(self, 
                 emission: Vector3 = None,
                 strength: float = 1.0,
                 temperature: float = 6500.0):
        
        if emission is None:
            # Convert color temperature to RGB (simplified)
            if temperature <= 1000:
                emission = Vector3(1.0, 0.2, 0.1)  # Warm
            elif temperature <= 4000:
                emission = Vector3(1.0, 0.6, 0.4)  # Neutral
            else:
                emission = Vector3(1.0, 0.9, 0.8)  # Cool
            
            emission = emission * strength
        
        super().__init__(
            albedo=Vector3(0, 0, 0),
            emission=emission * strength,
            metallic=0.0,
            roughness=1.0
        )
        self.strength = strength
        self.temperature = temperature

class MaterialLibrary:
    """Predefined material library for common materials"""
    
    @staticmethod
    def gold(roughness: float = 0.1) -> MetalMaterial:
        return MetalMaterial(
            albedo=Vector3(1.0, 0.86, 0.57),
            roughness=roughness,
            f0=Vector3(1.0, 0.86, 0.57)
        )
    
    @staticmethod
    def copper(roughness: float = 0.1) -> MetalMaterial:
        return MetalMaterial(
            albedo=Vector3(0.95, 0.64, 0.54),
            roughness=roughness,
            f0=Vector3(0.95, 0.64, 0.54)
        )
    
    @staticmethod
    def aluminum(roughness: float = 0.1) -> MetalMaterial:
        return MetalMaterial(
            albedo=Vector3(0.91, 0.92, 0.92),
            roughness=roughness,
            f0=Vector3(0.91, 0.92, 0.92)
        )
    
    @staticmethod
    def plastic(color: Vector3, roughness: float = 0.3) -> Material:
        return Material(
            albedo=color,
            metallic=0.0,
            roughness=roughness,
            specular=0.5
        )
    
    @staticmethod
    def rubber(color: Vector3, roughness: float = 0.8) -> Material:
        return Material(
            albedo=color,
            metallic=0.0,
            roughness=roughness,
            specular=0.1
        )
    
    @staticmethod
    def water(roughness: float = 0.05) -> GlassMaterial:
        return GlassMaterial(ior=1.33, roughness=roughness)
    
    @staticmethod
    def diamond(roughness: float = 0.01) -> GlassMaterial:
        return GlassMaterial(ior=2.42, roughness=roughness)
    
    @staticmethod
    def light(strength: float = 5.0) -> EmissiveMaterial:
        return EmissiveMaterial(emission=Vector3(1, 1, 1), strength=strength)

# Utility functions
def lerp_material(a: Material, b: Material, t: float) -> Material:
    """Linearly interpolate between two materials"""
    t = max(0.0, min(1.0, t))
    return Material(
        albedo=a.albedo * (1-t) + b.albedo * t,
        metallic=a.metallic * (1-t) + b.metallic * t,
        roughness=a.roughness * (1-t) + b.roughness * t,
        emission=a.emission * (1-t) + b.emission * t,
        ior=a.ior * (1-t) + b.ior * t,
        transmission=a.transmission * (1-t) + b.transmission * t,
        specular=a.specular * (1-t) + b.specular * t
    )

def create_material_from_dict(params: dict) -> Material:
    """Create material from dictionary parameters"""
    material_type = params.get('type', 'standard')
    
    if material_type == 'metal':
        return MetalMaterial(
            albedo=Vector3(*params.get('albedo', [0.9, 0.9, 0.9])),
            roughness=params.get('roughness', 0.1),
            f0=Vector3(*params.get('f0', [0.9, 0.9, 0.9])),
            anisotropic=params.get('anisotropic', 0.0)
        )
    elif material_type == 'glass':
        return GlassMaterial(
            ior=params.get('ior', 1.5),
            roughness=params.get('roughness', 0.0),
            tint=Vector3(*params.get('tint', [1, 1, 1]))
        )
    elif material_type == 'emissive':
        return EmissiveMaterial(
            emission=Vector3(*params.get('emission', [1, 1, 1])),
            strength=params.get('strength', 1.0)
        )
    else:  # standard
        return Material(
            albedo=Vector3(*params.get('albedo', [0.8, 0.8, 0.8])),
            metallic=params.get('metallic', 0.0),
            roughness=params.get('roughness', 0.5),
            emission=Vector3(*params.get('emission', [0, 0, 0])),
            ior=params.get('ior', 1.5),
            transmission=params.get('transmission', 0.0),
            specular=params.get('specular', 0.5)
        )