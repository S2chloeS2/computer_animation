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
import numpy as np

from ..core.types import Axis, nparray
from ..geometry.types import GeoType
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

        # NOTE: for now, all the shapes are assumed to be fixed.
        self.shape_type: list[GeoType] | None = None
        """Shape type, shape [shape_count], GeoType."""
        self.shape_transform: nparray | None = None
        """Shape transform, shape [shape_count, 4x4], float."""
        self.shape_inv_transform: nparray | None = None
        """Shape inverse transform, shape [shape_count, 4x4], float."""
        self.shape_penalty_params: nparray | None = None
        """Shape penalty parameters, shape [shape_count, 2], float.
        The first element is the stiffness, the second element is the damping.
        """
        self.shape_restitution_coeff: nparray | None = None
        """Shape restitution coefficient, shape [shape_count], float."""

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
        return s
