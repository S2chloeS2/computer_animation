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
        return s

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
