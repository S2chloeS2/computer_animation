#
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Written by Changxi Zheng <cxz@cs.columbia.edu>, 2026
#
from typing import Any

import fcl
import numpy as np
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

from ..core.types import Axis, nparray
from ..geometry.types import GeoType, ShapeFlags
from .state import State


class Model:
    def __init__(self):
        self.gravity = np.zeros(3, dtype=np.float64)
        """gravitational constant, which could be along any direction."""

        self.up_axis = Axis.Z
        """Initialize a Model object."""
        self.particle_count = 0
        """Total number of particles in the system."""
        self.particle_q: nparray | None = None
        """Particle positions, shape [particle_count, 3], float."""
        self.particle_qd: nparray | None = None
        """Particle velocities, shape [particle_count, 3], float."""
        self.particle_mass: nparray | None = None
        """Particle mass, shape [particle_count], float."""
        self.particle_inv_mass: nparray | None = None
        """Particle inverse mass, shape [particle_count], float."""
        self.particle_radius: nparray | None = None
        """Particle radius, shape [particle_count], float."""
        self.particle_flags: nparray | None = None
        """Particle enabled state, shape [particle_count], int."""
        self.particle_drag: nparray | None = None
        """Particle drag coefficient, shape [particle_count], float."""
        self.particle_restitution_coeff: nparray | None = None
        """Particle restitution coefficient, shape [particle_count], float."""
        self.particle_uv: nparray | None = None
        """Particle UV coordinates, shape [particle_count, 2], float."""

        self.spring_indices: nparray | None = None
        """Particle spring indices, shape [spring_count, 2], int."""
        self.spring_rest_length: nparray | None = None
        """Particle spring rest length, shape [spring_count], float."""
        self.spring_stiffness: nparray | None = None
        """Particle spring stiffness, shape [spring_count], float."""
        self.spring_damping: nparray | None = None
        """Particle spring damping, shape [spring_count], float."""

        self.gravitational_pairs: nparray | None = None
        """Gravitational pairs, shape [gravitational_count, 2], int."""
        self.gravitational_constant: nparray | None = None
        """Gravitational constant, shape [gravitational_count], float."""

        # NOTE: for now, all the PLANEs are assumed to be fixed.
        # NOTE: all shape's CoMs are assumed to be at the origin of its local frame.
        self.shape_type: list[GeoType] | None = None
        """Shape type, shape [shape_count], GeoType."""
        self.shape_flags: nparray | None = None
        """Shape enabled state, shape [shape_count], int."""
        self.shape_transform: nparray | None = None
        """Shape initial transform, shape [shape_count, 4x4], float."""
        self.shape_inv_transform: nparray | None = None
        """Shape inverse initial transform, shape [shape_count, 4x4], float."""
        self.shape_penalty_params: nparray | None = None
        """Shape penalty parameters, shape [shape_count, 2], float.
        The first element is the stiffness, the second element is the damping.
        """
        self.shape_mass: nparray | None = None
        """Shape mass, shape [particle_count], float."""
        self.shape_inv_mass: nparray | None = None
        """Shape inverse of mass, shape [particle_count], float."""
        self.shape_inertia: nparray | None = None
        """Shape's moment of inertia, shape [shape_count, 3x3], float."""
        self.shape_inv_inertia: nparray | None = None
        """Shape's inverse moment of inertia, shape [shape_count, 3x3], float."""
        self.shape_restitution_coeff: nparray | None = None
        """Shape restitution coefficient, shape [shape_count], float."""
        self.shape_fcl_geoms: list[Any] | None = None
        """Shape collision geometries, typically a fcl.shape (e.g., fcl.Sphere, fcl.Box, fcl.Halfspace) [shape_count]"""
        self.shape_collision_objects: list[fcl.CollisionObject] = []
        """FCL collision objects for collision detection, shape [shape_count], fcl.CollisionObject."""
        self.shape_meshes: list[Trimesh | None] = []
        """Shape meshes, shape [shape_count], Trimesh."""

        # -------------------------------------------------------------
        # for cloth sim.
        # Cloth has three types of contraints: stretch, shear, and bending.
        # Material parameters for stretch and shaar are stored in the tri_materials array.
        # Material parameters for bending are stored in the edge_materials array.
        self.tri_indices: nparray | None = None
        """Triangle element indices, shape [tri_count, 3], int."""
        self.tri_areas: nparray | None = None
        """Triangle element areas, shape [tri_count], float."""
        self.tri_uv_inv: nparray | None = None
        """Triangle element UV-coordinate inverse, shape [tri_count, 2, 2], float.
        This will be used to compute the deformation gradient of the triangle element.
        """
        self.tri_materials: nparray | None = None
        """Triangle materials, shape [tri_count, 4], float.
        The first element is the stretch stiffness, the second element is the stretch damping,
        the third element is the shear stiffness, the fourth element is the shear damping.
        """

        self.edge_indices: nparray | None = None
        """Bending edge indices, shape [edge_count, 4], int, each row is [o0, o1, v1, v2],
           where o1, o2 are on the edge, v1 is on the left side of the edge, and v2 is on
           the right side of the edge.
        """
        self.edge_rest_angle: nparray | None = None
        """Bending edge rest angle, shape [edge_count], float."""
        self.edge_materials: nparray | None = None
        """Bending edge stiffness and damping, shape [edge_count, 2], float."""

        # -------------------------------------------------------------
        # parameters for fluid sim
        self.init_fluid_map: list[str] | None = None
        """A list of strings to represent the initial fluid setup.
        Each string is a row of the fluid map, 'o' indicates an water voxel, and '.' indicates an air voxel.
        For example,

        oooo....
        oooo....
        oooo....

        indicates a 3x8 grid in which the first 4 columns are water voxels.
        """
        self.fluid_domain_size: nparray | None = None
        """Fluid domain size, shape [3 or 2], float.
        If 3, the domain is a cube (3D sim).
        If 2, the domain is a rectangle (2D sim).
        """
        self.fluid_domain_res: nparray | None = None
        """Fluid domain resolution, shape [3 or 2], uint32.
        NOTE:
          fluid_domain_res[0] is the number of cells in the x direction,
          fluid_domain_res[1] is the number of cells in the y direction.
        """
        self.fluid_cell_size: float = 0.0
        """Fluid cell size, float.
        The following relation must hold:
            fluid_domain_size = fluid_domain_res * fluid_cell_size
        """

    @property
    def edge_count(self) -> int:
        """
        The number of bending edges in the model.
        """
        return 0 if self.edge_indices is None else self.edge_indices.shape[0]

    @property
    def spring_count(self) -> int:
        """
        The number of springs in the model.
        """
        return 0 if self.spring_rest_length is None else len(self.spring_rest_length)

    @property
    def shape_count(self) -> int:
        """
        The number of shapes in the model.
        """
        return 0 if self.shape_type is None else len(self.shape_type)

    @property
    def tri_count(self) -> int:
        """
        The number of triangles in the model.
        """
        return 0 if self.tri_indices is None else self.tri_indices.shape[0]

    @property
    def gravitational_count(self) -> int:
        """
        The number of gravitational pairs in the model.
        """
        return 0 if self.gravitational_constant is None else len(self.gravitational_constant)

    def state(self) -> State:
        s = State()
        # particles
        if self.particle_count:
            s.particle_q = self.particle_q.copy()
            s.particle_qd = self.particle_qd.copy()
            s.particle_f = np.zeros_like(self.particle_qd)
        # shapes
        if self.shape_count:
            s.shape_q = np.empty((self.shape_count, 7), dtype=self.shape_transform.dtype)
            s.shape_qd = np.zeros((self.shape_count, 6), dtype=self.shape_transform.dtype)
            s.shape_f = np.zeros_like(s.shape_qd)
            # set the shape_q
            for i in range(self.shape_count):
                if self.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                    s.shape_q[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=self.shape_transform.dtype)
                else:
                    s.shape_q[i, :3] = self.shape_transform[i][:3, 3]
                    s.shape_q[i, 3:] = R.from_matrix(self.shape_transform[i][:3, :3]).as_quat(scalar_first=True)
        # fluid
        if self.fluid_domain_res is not None:
            s.fluid_u_x = np.zeros((self.fluid_domain_res[1], self.fluid_domain_res[0] + 1), dtype=np.float64)
            s.fluid_u_y = np.zeros((self.fluid_domain_res[1] + 1, self.fluid_domain_res[0]), dtype=np.float64)
            self.initialize_fluid_particles(s)
        return s

    def initialize_fluid_particles(self, state: State) -> None:
        """
        Initialize the fluid particles. Sample particles according to the initial fluid map.
        """
        ps = []
        dx = self.fluid_cell_size
        for j, row in enumerate(reversed(self.init_fluid_map)):
            for i, x in enumerate(row):
                if x == "o":
                    sx = i * self.fluid_cell_size
                    sy = j * self.fluid_cell_size
                    ps.append([sx + dx / 2, sy + dx / 2])
                    ps.append([sx + dx / 4, sy + dx / 4])
                    ps.append([sx + dx * 3.0 / 4, sy + dx / 4])
                    ps.append([sx + dx / 4, sy + dx * 3.0 / 4])
                    ps.append([sx + dx * 3.0 / 4, sy + dx * 3.0 / 4])

        state.fluid_particle_q = np.asarray(ps, dtype=np.float64)
        state.fluid_particle_qd = np.zeros_like(state.fluid_particle_q)

    def initialize_collision_objects(self) -> None:
        """
        Initialize the collision objects in the model
        """
        self.shape_collision_objects = []
        for i in range(self.shape_count):
            self.shape_collision_objects.append(fcl.CollisionObject(self.shape_fcl_geoms[i]))

            if self.shape_type[i] != GeoType.HALF_SPACE_PLANE and self.shape_flags[i] & ShapeFlags.ACTIVE.value == 0:
                # for fixed shapes set the transform
                self.shape_collision_objects[i].setTransform(
                    fcl.Transform(
                        self.shape_transform[i][:3, :3],
                        self.shape_transform[i][:3, 3],
                    ),
                )
                # For half space planes, the transform has already backed in the fcl.Halfspace object
                # (see builder.py:add_shape_half_space_plane)
                # So we don't need to set the transform again.

    def update_collision_objects(self, state: State) -> None:
        """
        Update the collision objects transformations in the model using the given state
        """
        for i in range(self.shape_count):
            # only update the transform for active shapes
            if self.shape_flags[i] & ShapeFlags.ACTIVE.value != 0:
                self.shape_collision_objects[i].setTransform(
                    fcl.Transform(
                        state.shape_q[i, 3:],  # rotation as quaternion
                        state.shape_q[i, :3],
                    ),
                )
