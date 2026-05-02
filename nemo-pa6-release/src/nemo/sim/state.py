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
from ..core.types import nparray


class State:
    def __init__(self) -> None:
        """
        Initialize an empty State object.
        To ensure that the attributes are properly allocated create the State object via :meth:`newton.Model.state`
        instead.
        """

        # ------------------------------------------------------------
        # particles
        self.particle_q: nparray | None = None
        """3D positions of particles, shape (particle_count, 3)"""
        self.particle_qd: nparray | None = None
        """3D velocities of particles, shape (particle_count, 3)."""
        self.particle_f: nparray | None = None
        """3D forces on particles, shape (particle_count, 3)."""

        # ------------------------------------------------------------
        # shapes
        self.shape_q: nparray | None = None
        """3D positions of shapes, shape (shape_count, 7)
        The first 3 elements are the position of the shape's center of mass,
        The last 4 elements are the quaternion (w, x, y, z) of the shape's orientation
        (scalar first).
        """
        self.shape_qd: nparray | None = None
        """3D velocities of shapes, shape (shape_count, 6)
        The first 3 elements are the linear velocity of the shape's center of mass,
        The last 3 elements are the angular velocity of the shape's center of mass.
        """
        self.shape_f: nparray | None = None
        """3D forces on shapes, shape (shape_count, 6)
        The first 3 elements are the linear force on the shape's center of mass,
        The last 3 elements are the torque on the shape's center of mass.
        """

        # ------------------------------------------------------------
        # fluid
        # Assuming the fluid grid has a resolution of N x M
        self.fluid_particle_q: nparray | None = None
        """3D positions of fluid particles, shape (particle_count, 2)"""
        self.fluid_particle_qd: nparray | None = None
        """3D velocities of fluid particles, shape (particle_count, 2)"""
        self.fluid_u_x: nparray | None = None
        """flud velocity in x direction, shape (N, M+1)
        u_x are defined at the center of the vertical grid edges.
        Note that on a N x M grid, there are M+1 vertical grid edges.
        """
        self.fluid_u_y: nparray | None = None
        """flud velocity in y direction, shape (N+1, M)
        u_y are defined at the center of the horizontal grid edges
        Note that on a N x M grid, there are N+1 horizontal grid edges.
        """

    def clear_forces(self) -> None:
        """
        Clear all force arrays (for particles and bodies) in the state object.

        Sets all entries of :attr:`particle_f` and :attr:`body_f` to zero, if present.
        """
        if self.particle_f is not None:
            self.particle_f.fill(0)
        if self.shape_f is not None:
            self.shape_f.fill(0)
