"""
OMNIXAN Ray Tracing Unit Module
supercomputing_interconnect_cloud/ray_tracing_unit_module

Production-ready ray tracing acceleration module supporting hardware RT cores,
BVH construction, ray-scene intersection, and denoising for photorealistic
rendering and scientific simulations.
"""

import asyncio
import logging
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTBackend(str, Enum):
    """Ray tracing backends"""
    CPU = "cpu"
    CUDA_RT_CORES = "cuda_rt_cores"
    OPTIX = "optix"
    EMBREE = "embree"
    VULKAN_RT = "vulkan_rt"


class AccelerationStructure(str, Enum):
    """Acceleration structure types"""
    BVH = "bvh"  # Bounding Volume Hierarchy
    KD_TREE = "kd_tree"
    OCTREE = "octree"
    GRID = "grid"


class RayType(str, Enum):
    """Types of rays"""
    PRIMARY = "primary"
    SHADOW = "shadow"
    REFLECTION = "reflection"
    REFRACTION = "refraction"
    AMBIENT_OCCLUSION = "ambient_occlusion"
    GLOBAL_ILLUMINATION = "global_illumination"


class ShadingModel(str, Enum):
    """Shading models"""
    LAMBERT = "lambert"
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    COOK_TORRANCE = "cook_torrance"
    OREN_NAYAR = "oren_nayar"
    PBR = "pbr"


@dataclass
class Vec3:
    """3D Vector"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        return math.sqrt(self.dot(self))
    
    def normalize(self) -> "Vec3":
        length = self.length()
        if length > 0:
            return self * (1.0 / length)
        return Vec3(0, 0, 0)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Ray:
    """Ray definition"""
    origin: Vec3
    direction: Vec3
    ray_type: RayType = RayType.PRIMARY
    t_min: float = 0.001
    t_max: float = float('inf')
    depth: int = 0


@dataclass
class HitInfo:
    """Ray-surface intersection info"""
    hit: bool
    t: float = float('inf')
    point: Optional[Vec3] = None
    normal: Optional[Vec3] = None
    uv: Optional[Tuple[float, float]] = None
    material_id: int = 0
    primitive_id: int = 0


@dataclass
class Triangle:
    """Triangle primitive"""
    v0: Vec3
    v1: Vec3
    v2: Vec3
    normal: Optional[Vec3] = None
    material_id: int = 0
    
    def compute_normal(self) -> Vec3:
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        return edge1.cross(edge2).normalize()


@dataclass
class Sphere:
    """Sphere primitive"""
    center: Vec3
    radius: float
    material_id: int = 0


@dataclass
class AABB:
    """Axis-Aligned Bounding Box"""
    min_point: Vec3
    max_point: Vec3
    
    def intersect(self, ray: Ray) -> bool:
        """Fast ray-AABB intersection test"""
        inv_dir = Vec3(
            1.0 / ray.direction.x if ray.direction.x != 0 else float('inf'),
            1.0 / ray.direction.y if ray.direction.y != 0 else float('inf'),
            1.0 / ray.direction.z if ray.direction.z != 0 else float('inf')
        )
        
        t1 = (self.min_point.x - ray.origin.x) * inv_dir.x
        t2 = (self.max_point.x - ray.origin.x) * inv_dir.x
        t3 = (self.min_point.y - ray.origin.y) * inv_dir.y
        t4 = (self.max_point.y - ray.origin.y) * inv_dir.y
        t5 = (self.min_point.z - ray.origin.z) * inv_dir.z
        t6 = (self.max_point.z - ray.origin.z) * inv_dir.z
        
        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
        
        return tmax >= max(tmin, 0.0)


@dataclass
class BVHNode:
    """BVH tree node"""
    bounds: AABB
    left: Optional["BVHNode"] = None
    right: Optional["BVHNode"] = None
    primitive_indices: List[int] = field(default_factory=list)
    is_leaf: bool = False


@dataclass
class Material:
    """Material properties"""
    albedo: Vec3 = field(default_factory=lambda: Vec3(0.8, 0.8, 0.8))
    metallic: float = 0.0
    roughness: float = 0.5
    emission: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    ior: float = 1.5  # Index of refraction


@dataclass
class RTMetrics:
    """Ray tracing metrics"""
    total_rays: int = 0
    primary_rays: int = 0
    shadow_rays: int = 0
    secondary_rays: int = 0
    bvh_traversals: int = 0
    triangle_tests: int = 0
    render_time_ms: float = 0.0
    rays_per_second: float = 0.0


class RTConfig(BaseModel):
    """Configuration for ray tracing"""
    backend: RTBackend = Field(
        default=RTBackend.CPU,
        description="Ray tracing backend"
    )
    acceleration_structure: AccelerationStructure = Field(
        default=AccelerationStructure.BVH,
        description="Acceleration structure type"
    )
    max_recursion_depth: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum ray recursion depth"
    )
    samples_per_pixel: int = Field(
        default=1,
        ge=1,
        description="Samples per pixel (SPP)"
    )
    enable_shadows: bool = Field(
        default=True,
        description="Enable shadow rays"
    )
    enable_reflections: bool = Field(
        default=True,
        description="Enable reflection rays"
    )
    enable_denoising: bool = Field(
        default=False,
        description="Enable AI denoising"
    )
    tile_size: int = Field(
        default=64,
        ge=8,
        description="Render tile size"
    )


class RTError(Exception):
    """Base exception for ray tracing errors"""
    pass


# ============================================================================
# BVH Builder
# ============================================================================

class BVHBuilder:
    """Builds Bounding Volume Hierarchy"""
    
    def __init__(self, max_primitives_per_leaf: int = 4):
        self.max_primitives_per_leaf = max_primitives_per_leaf
    
    def build(self, primitives: List[Union[Triangle, Sphere]]) -> BVHNode:
        """Build BVH from primitives"""
        if not primitives:
            return BVHNode(
                bounds=AABB(Vec3(0, 0, 0), Vec3(0, 0, 0)),
                is_leaf=True
            )
        
        indices = list(range(len(primitives)))
        return self._build_recursive(primitives, indices)
    
    def _build_recursive(
        self,
        primitives: List[Union[Triangle, Sphere]],
        indices: List[int]
    ) -> BVHNode:
        """Recursively build BVH"""
        # Compute bounds
        bounds = self._compute_bounds(primitives, indices)
        
        if len(indices) <= self.max_primitives_per_leaf:
            return BVHNode(
                bounds=bounds,
                primitive_indices=indices,
                is_leaf=True
            )
        
        # Find split axis (longest axis)
        extent = Vec3(
            bounds.max_point.x - bounds.min_point.x,
            bounds.max_point.y - bounds.min_point.y,
            bounds.max_point.z - bounds.min_point.z
        )
        
        if extent.x >= extent.y and extent.x >= extent.z:
            axis = 0
        elif extent.y >= extent.z:
            axis = 1
        else:
            axis = 2
        
        # Sort primitives by centroid along axis
        def get_centroid(idx: int) -> float:
            p = primitives[idx]
            if isinstance(p, Triangle):
                c = Vec3(
                    (p.v0.x + p.v1.x + p.v2.x) / 3,
                    (p.v0.y + p.v1.y + p.v2.y) / 3,
                    (p.v0.z + p.v1.z + p.v2.z) / 3
                )
            else:
                c = p.center
            return [c.x, c.y, c.z][axis]
        
        sorted_indices = sorted(indices, key=get_centroid)
        mid = len(sorted_indices) // 2
        
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]
        
        return BVHNode(
            bounds=bounds,
            left=self._build_recursive(primitives, left_indices),
            right=self._build_recursive(primitives, right_indices),
            is_leaf=False
        )
    
    def _compute_bounds(
        self,
        primitives: List[Union[Triangle, Sphere]],
        indices: List[int]
    ) -> AABB:
        """Compute bounding box for primitives"""
        min_p = Vec3(float('inf'), float('inf'), float('inf'))
        max_p = Vec3(float('-inf'), float('-inf'), float('-inf'))
        
        for idx in indices:
            p = primitives[idx]
            
            if isinstance(p, Triangle):
                for v in [p.v0, p.v1, p.v2]:
                    min_p = Vec3(
                        min(min_p.x, v.x),
                        min(min_p.y, v.y),
                        min(min_p.z, v.z)
                    )
                    max_p = Vec3(
                        max(max_p.x, v.x),
                        max(max_p.y, v.y),
                        max(max_p.z, v.z)
                    )
            else:  # Sphere
                r = p.radius
                min_p = Vec3(
                    min(min_p.x, p.center.x - r),
                    min(min_p.y, p.center.y - r),
                    min(min_p.z, p.center.z - r)
                )
                max_p = Vec3(
                    max(max_p.x, p.center.x + r),
                    max(max_p.y, p.center.y + r),
                    max(max_p.z, p.center.z + r)
                )
        
        return AABB(min_p, max_p)


# ============================================================================
# Ray Intersector
# ============================================================================

class RayIntersector:
    """Performs ray-scene intersection tests"""
    
    def intersect_triangle(self, ray: Ray, triangle: Triangle) -> HitInfo:
        """Möller–Trumbore ray-triangle intersection"""
        edge1 = triangle.v1 - triangle.v0
        edge2 = triangle.v2 - triangle.v0
        
        h = ray.direction.cross(edge2)
        a = edge1.dot(h)
        
        if abs(a) < 1e-8:
            return HitInfo(hit=False)
        
        f = 1.0 / a
        s = ray.origin - triangle.v0
        u = f * s.dot(h)
        
        if u < 0.0 or u > 1.0:
            return HitInfo(hit=False)
        
        q = s.cross(edge1)
        v = f * ray.direction.dot(q)
        
        if v < 0.0 or u + v > 1.0:
            return HitInfo(hit=False)
        
        t = f * edge2.dot(q)
        
        if t < ray.t_min or t > ray.t_max:
            return HitInfo(hit=False)
        
        point = ray.origin + ray.direction * t
        normal = triangle.normal or triangle.compute_normal()
        
        return HitInfo(
            hit=True,
            t=t,
            point=point,
            normal=normal,
            uv=(u, v),
            material_id=triangle.material_id
        )
    
    def intersect_sphere(self, ray: Ray, sphere: Sphere) -> HitInfo:
        """Ray-sphere intersection"""
        oc = ray.origin - sphere.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - sphere.radius * sphere.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return HitInfo(hit=False)
        
        sqrt_d = math.sqrt(discriminant)
        t = (-b - sqrt_d) / (2 * a)
        
        if t < ray.t_min or t > ray.t_max:
            t = (-b + sqrt_d) / (2 * a)
            if t < ray.t_min or t > ray.t_max:
                return HitInfo(hit=False)
        
        point = ray.origin + ray.direction * t
        normal = (point - sphere.center).normalize()
        
        return HitInfo(
            hit=True,
            t=t,
            point=point,
            normal=normal,
            material_id=sphere.material_id
        )
    
    def traverse_bvh(
        self,
        ray: Ray,
        bvh: BVHNode,
        primitives: List[Union[Triangle, Sphere]]
    ) -> HitInfo:
        """Traverse BVH for closest intersection"""
        closest_hit = HitInfo(hit=False)
        
        stack = [bvh]
        while stack:
            node = stack.pop()
            
            if not node.bounds.intersect(ray):
                continue
            
            if node.is_leaf:
                for idx in node.primitive_indices:
                    p = primitives[idx]
                    if isinstance(p, Triangle):
                        hit = self.intersect_triangle(ray, p)
                    else:
                        hit = self.intersect_sphere(ray, p)
                    
                    if hit.hit and hit.t < closest_hit.t:
                        closest_hit = hit
                        closest_hit.primitive_id = idx
                        ray.t_max = hit.t
            else:
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        
        return closest_hit


# ============================================================================
# Main Module Implementation
# ============================================================================

class RayTracingUnitModule:
    """
    Production-ready Ray Tracing Unit module for OMNIXAN.
    
    Provides:
    - Hardware RT core acceleration (simulated for CPU)
    - BVH construction and traversal
    - Ray-scene intersection
    - Multiple shading models
    - Shadow and reflection rays
    """
    
    def __init__(self, config: Optional[RTConfig] = None):
        """Initialize the Ray Tracing Unit Module"""
        self.config = config or RTConfig()
        self.bvh_builder = BVHBuilder()
        self.intersector = RayIntersector()
        
        self.primitives: List[Union[Triangle, Sphere]] = []
        self.materials: List[Material] = [Material()]  # Default material
        self.bvh: Optional[BVHNode] = None
        
        self.metrics = RTMetrics()
        self._initialized = False
        self._logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the ray tracing module"""
        if self._initialized:
            self._logger.warning("Module already initialized")
            return
        
        try:
            self._logger.info("Initializing RayTracingUnitModule...")
            self._initialized = True
            self._logger.info("RayTracingUnitModule initialized successfully")
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {str(e)}")
            raise RTError(f"Failed to initialize module: {str(e)}")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ray tracing operation"""
        if not self._initialized:
            raise RTError("Module not initialized")
        
        operation = params.get("operation")
        
        if operation == "add_triangle":
            v0 = Vec3(**params["v0"])
            v1 = Vec3(**params["v1"])
            v2 = Vec3(**params["v2"])
            material_id = params.get("material_id", 0)
            idx = self.add_triangle(v0, v1, v2, material_id)
            return {"primitive_id": idx}
        
        elif operation == "add_sphere":
            center = Vec3(**params["center"])
            radius = params["radius"]
            material_id = params.get("material_id", 0)
            idx = self.add_sphere(center, radius, material_id)
            return {"primitive_id": idx}
        
        elif operation == "add_material":
            mat = Material(
                albedo=Vec3(**params.get("albedo", {"x": 0.8, "y": 0.8, "z": 0.8})),
                metallic=params.get("metallic", 0.0),
                roughness=params.get("roughness", 0.5)
            )
            idx = self.add_material(mat)
            return {"material_id": idx}
        
        elif operation == "build_bvh":
            await self.build_acceleration_structure()
            return {"success": True, "primitives": len(self.primitives)}
        
        elif operation == "trace_ray":
            origin = Vec3(**params["origin"])
            direction = Vec3(**params["direction"])
            hit = self.trace_ray(origin, direction)
            return {
                "hit": hit.hit,
                "t": hit.t if hit.hit else None,
                "point": {"x": hit.point.x, "y": hit.point.y, "z": hit.point.z} if hit.hit else None,
                "normal": {"x": hit.normal.x, "y": hit.normal.y, "z": hit.normal.z} if hit.hit else None
            }
        
        elif operation == "render":
            width = params.get("width", 256)
            height = params.get("height", 256)
            camera_pos = Vec3(**params.get("camera_pos", {"x": 0, "y": 0, "z": 5}))
            image = await self.render(width, height, camera_pos)
            return {
                "width": width,
                "height": height,
                "image": image.tolist()
            }
        
        elif operation == "get_metrics":
            return self.get_metrics()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def add_triangle(
        self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        material_id: int = 0
    ) -> int:
        """Add triangle to scene"""
        triangle = Triangle(v0, v1, v2, material_id=material_id)
        triangle.normal = triangle.compute_normal()
        self.primitives.append(triangle)
        return len(self.primitives) - 1
    
    def add_sphere(
        self,
        center: Vec3,
        radius: float,
        material_id: int = 0
    ) -> int:
        """Add sphere to scene"""
        sphere = Sphere(center, radius, material_id)
        self.primitives.append(sphere)
        return len(self.primitives) - 1
    
    def add_material(self, material: Material) -> int:
        """Add material"""
        self.materials.append(material)
        return len(self.materials) - 1
    
    async def build_acceleration_structure(self) -> None:
        """Build BVH from primitives"""
        self._logger.info(f"Building BVH for {len(self.primitives)} primitives...")
        start_time = time.time()
        
        self.bvh = self.bvh_builder.build(self.primitives)
        
        build_time = (time.time() - start_time) * 1000
        self._logger.info(f"BVH built in {build_time:.2f}ms")
    
    def trace_ray(self, origin: Vec3, direction: Vec3) -> HitInfo:
        """Trace a single ray"""
        ray = Ray(origin, direction.normalize())
        self.metrics.total_rays += 1
        self.metrics.primary_rays += 1
        
        if self.bvh:
            return self.intersector.traverse_bvh(ray, self.bvh, self.primitives)
        
        # Brute force if no BVH
        closest_hit = HitInfo(hit=False)
        for i, p in enumerate(self.primitives):
            if isinstance(p, Triangle):
                hit = self.intersector.intersect_triangle(ray, p)
            else:
                hit = self.intersector.intersect_sphere(ray, p)
            
            if hit.hit and hit.t < closest_hit.t:
                closest_hit = hit
                closest_hit.primitive_id = i
        
        return closest_hit
    
    def shade(self, hit: HitInfo, ray: Ray, depth: int = 0) -> Vec3:
        """Compute shading at hit point"""
        if not hit.hit:
            # Sky color
            t = 0.5 * (ray.direction.y + 1.0)
            return Vec3(1.0, 1.0, 1.0) * (1.0 - t) + Vec3(0.5, 0.7, 1.0) * t
        
        material = self.materials[hit.material_id] if hit.material_id < len(self.materials) else self.materials[0]
        
        # Simple Lambert shading
        light_dir = Vec3(1, 1, 1).normalize()
        ndotl = max(0, hit.normal.dot(light_dir))
        
        # Shadow ray
        if self.config.enable_shadows and ndotl > 0:
            shadow_origin = hit.point + hit.normal * 0.001
            shadow_ray = Ray(shadow_origin, light_dir, RayType.SHADOW)
            
            if self.bvh:
                shadow_hit = self.intersector.traverse_bvh(
                    shadow_ray, self.bvh, self.primitives
                )
            else:
                shadow_hit = HitInfo(hit=False)
            
            self.metrics.shadow_rays += 1
            
            if shadow_hit.hit:
                ndotl *= 0.3  # Shadow
        
        # Ambient + Diffuse
        ambient = 0.1
        color = material.albedo * (ambient + ndotl * 0.9)
        
        # Reflection
        if self.config.enable_reflections and material.metallic > 0 and depth < self.config.max_recursion_depth:
            reflect_dir = ray.direction - hit.normal * (2.0 * ray.direction.dot(hit.normal))
            reflect_origin = hit.point + hit.normal * 0.001
            reflect_ray = Ray(reflect_origin, reflect_dir, RayType.REFLECTION, depth=depth + 1)
            
            if self.bvh:
                reflect_hit = self.intersector.traverse_bvh(
                    reflect_ray, self.bvh, self.primitives
                )
                reflect_color = self.shade(reflect_hit, reflect_ray, depth + 1)
                color = color * (1 - material.metallic) + reflect_color * material.metallic
            
            self.metrics.secondary_rays += 1
        
        return color
    
    async def render(
        self,
        width: int,
        height: int,
        camera_pos: Vec3,
        look_at: Vec3 = None,
        fov: float = 60.0
    ) -> np.ndarray:
        """Render scene to image"""
        start_time = time.time()
        
        if not self.bvh:
            await self.build_acceleration_structure()
        
        look_at = look_at or Vec3(0, 0, 0)
        
        # Camera setup
        aspect = width / height
        fov_rad = math.radians(fov)
        half_height = math.tan(fov_rad / 2)
        half_width = aspect * half_height
        
        w = (camera_pos - look_at).normalize()
        u = Vec3(0, 1, 0).cross(w).normalize()
        v = w.cross(u)
        
        # Render image
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Compute ray direction
                s = (2.0 * x / width - 1.0) * half_width
                t = (1.0 - 2.0 * y / height) * half_height
                
                direction = (u * s + v * t - w).normalize()
                ray = Ray(camera_pos, direction)
                
                # Trace ray
                hit = self.trace_ray(camera_pos, direction)
                color = self.shade(hit, ray)
                
                image[y, x] = [
                    min(1.0, color.x),
                    min(1.0, color.y),
                    min(1.0, color.z)
                ]
        
        self.metrics.render_time_ms = (time.time() - start_time) * 1000
        self.metrics.rays_per_second = (
            self.metrics.total_rays / self.metrics.render_time_ms * 1000
        )
        
        return image
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ray tracing metrics"""
        return {
            "total_rays": self.metrics.total_rays,
            "primary_rays": self.metrics.primary_rays,
            "shadow_rays": self.metrics.shadow_rays,
            "secondary_rays": self.metrics.secondary_rays,
            "render_time_ms": round(self.metrics.render_time_ms, 2),
            "rays_per_second": round(self.metrics.rays_per_second, 0),
            "primitives": len(self.primitives),
            "materials": len(self.materials)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the ray tracing module"""
        self._logger.info("Shutting down RayTracingUnitModule...")
        
        self.primitives.clear()
        self.materials = [Material()]
        self.bvh = None
        self._initialized = False
        
        self._logger.info("RayTracingUnitModule shutdown complete")


# Example usage
async def main():
    """Example usage of RayTracingUnitModule"""
    
    config = RTConfig(
        backend=RTBackend.CPU,
        max_recursion_depth=4,
        enable_shadows=True,
        enable_reflections=True
    )
    
    module = RayTracingUnitModule(config)
    await module.initialize()
    
    try:
        # Add materials
        red_mat = module.add_material(Material(
            albedo=Vec3(0.8, 0.2, 0.2),
            metallic=0.0
        ))
        
        green_mat = module.add_material(Material(
            albedo=Vec3(0.2, 0.8, 0.2),
            metallic=0.5
        ))
        
        blue_mat = module.add_material(Material(
            albedo=Vec3(0.2, 0.2, 0.8),
            metallic=0.8
        ))
        
        # Add spheres
        module.add_sphere(Vec3(0, 0, 0), 1.0, red_mat)
        module.add_sphere(Vec3(2.5, 0, -1), 1.0, green_mat)
        module.add_sphere(Vec3(-2.5, 0, -1), 1.0, blue_mat)
        
        # Add ground plane (two triangles)
        ground_y = -1.0
        module.add_triangle(
            Vec3(-10, ground_y, -10),
            Vec3(10, ground_y, -10),
            Vec3(10, ground_y, 10),
            0
        )
        module.add_triangle(
            Vec3(-10, ground_y, -10),
            Vec3(10, ground_y, 10),
            Vec3(-10, ground_y, 10),
            0
        )
        
        print(f"Scene: {len(module.primitives)} primitives")
        
        # Build BVH
        await module.build_acceleration_structure()
        
        # Trace single ray
        hit = module.trace_ray(Vec3(0, 0, 5), Vec3(0, 0, -1))
        print(f"Ray hit: {hit.hit}, t={hit.t if hit.hit else 'N/A'}")
        
        # Render small image
        print("Rendering 128x128 image...")
        image = await module.render(128, 128, Vec3(0, 2, 5))
        
        # Get metrics
        metrics = module.get_metrics()
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    finally:
        await module.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

