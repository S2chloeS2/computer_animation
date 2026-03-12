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
from scipy.spatial.transform import Rotation as R

from ..core.types import (
    Axis,
    AxisType,
    Vec3,
)
from ..geometry.types import GeoType, ParticleFlags
from .model import Model


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Similar to many popular simulation engine (such as Newton, Drake, Mujoco),
    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    represents the scene using standard Python data structures like lists,
    which are convenient but unsuitable for efficient simulation.
    Call :meth:`finalize` to construct a simulation-ready Model.
    """

    # -------------------------------------------------------------------------
    # Default parameters

    # Default particle settings
    DEFAULT_PARTICLE_RADIUS = 0.1
    # Default restitution coefficients
    DEFAULT_PARTICLE_RESTITUTION_COEFF = 0.5
    DEFAULT_SHAPE_RESTITUTION_COEFF = 0.5

    DEFAULT_SHAPE_PENALTY_STIFFNESS = 1000.0
    DEFAULT_SHAPE_PENALTY_DAMPING = 0.0
    # -------------------------------------------------------------------------

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):

        self.up_axis: Axis = Axis.from_any(up_axis)
        self.gravity: float = gravity

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_drag = []
        self.particle_restitution_coeff = []

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []

        # gravitational
        self.gravitational_pairs = []
        self.gravitational_constant = []

        # shapes
        self.shape_type = []  # list[GeoType]
        self.shape_transform = []  # nparray (shape_count, 4x4)
        self.shape_inv_transform = []  # nparray (shape_count, 4x4)
        self.shape_penalty_params = []  # nparray (shape_count, 2), float
        self.shape_restitution_coeff = []  # nparray (shape_count,), float

    @property
    def particle_count(self) -> int:
        """
        The number of particles in the model.
        """
        return len(self.particle_q)

    @property
    def spring_count(self) -> int:
        """
        The number of springs in the model.
        """
        return len(self.spring_rest_length)

    # particles
    def add_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        radius: float | None = None,
        drag: float | None = None,
        flags: int = ParticleFlags.ACTIVE.value,
        restitution_coeff: float | None = None,
    ) -> int:
        """Adds a single particle to the model.

        Args:
            pos: The initial position of the particle.
            vel: The initial velocity of the particle.
            mass: The mass of the particle.
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default
                    value (:attr:`DEFAULT_PARTICLE_RADIUS`).
            flags: The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.

        Returns:
            The index of the particle in the system.
        """
        if (isinstance(pos, list) and len(pos) != 3) or (isinstance(pos, np.ndarray) and pos.shape != (3,)):
            raise RuntimeError("pos must have a length of 3")
        if (isinstance(vel, list) and len(vel) != 3) or (isinstance(vel, np.ndarray) and vel.shape != (3,)):
            raise RuntimeError("vel must have a length of 3")
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        self.particle_flags.append(flags)
        if radius is None:
            self.particle_radius.append(self.DEFAULT_PARTICLE_RADIUS)
        else:
            self.particle_radius.append(radius)
        if drag is None:
            self.particle_drag.append(0.0)
        else:
            self.particle_drag.append(drag)
        if restitution_coeff is None:
            self.particle_restitution_coeff.append(self.DEFAULT_PARTICLE_RESTITUTION_COEFF)
        else:
            self.particle_restitution_coeff.append(restitution_coeff)
        return self.particle_count - 1

    def add_particles(
        self,
        pos: list[Vec3],
        vel: list[Vec3],
        mass: list[float],
        radius: list[float] | None = None,
        drag: list[float] | None = None,
        flags: list[int] | None = None,
        restitution_coeff: list[float] | None = None,
    ) -> None:
        """Adds a group particles to the model.

        Args:
            pos: The initial positions of the particle.
            vel: The initial velocities of the particle.
            mass: The mass of the particles.
            radius: The radius of the particles used in collision handling. If None, the radius is set to the default
            value (:attr:`DEFAULT_PARTICLE_RADIUS`).
            flags: The flags that control the dynamical behavior of the particles, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.
        """
        # check data
        for p, v in zip(pos, vel, strict=True):
            if (isinstance(p, list) and len(p) != 3) or (isinstance(p, np.ndarray) and p.shape != (3,)):
                raise RuntimeError("pos must have a length of 3")
            if (isinstance(v, list) and len(v) != 3) or (isinstance(v, np.ndarray) and v.shape != (3,)):
                raise RuntimeError("vel must have a length of 3")

        self.particle_q.extend(pos)
        self.particle_qd.extend(vel)
        self.particle_mass.extend(mass)
        if radius is None:
            radius = [self.DEFAULT_PARTICLE_RADIUS] * len(pos)
        if drag is None:
            drag = [0.0] * len(pos)
        if flags is None:
            flags = [ParticleFlags.ACTIVE.value] * len(pos)
        if restitution_coeff is None:
            restitution_coeff = [self.DEFAULT_PARTICLE_RESTITUTION_COEFF] * len(pos)
        self.particle_radius.extend(radius)
        self.particle_flags.extend(flags)
        self.particle_drag.extend(drag)
        self.particle_restitution_coeff.extend(restitution_coeff)

    def add_spring(self, i: int, j, ke: float, kd: float | None = None, rest_length: float | None = None):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            rest_length: The actuation level of the spring

        Note:
            If kd is None, zero damping will be used.

            If rest_length is None, the spring is created with a rest-length
            based on the distance between the particles in their initial
            configuration.
        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(0.0 if kd is None else kd)
        if rest_length is not None:
            if rest_length < 0:
                raise RuntimeError(f"Spring rest-length ({rest_length}) can't be negative")
            self.spring_rest_length.append(rest_length)
        else:
            self.spring_rest_length.append(-1.0)

    def add_shape_half_space_plane(
        self,
        plane_normal: Vec3,
        plane_offset: float,
        stiffness: float | None = None,
        damping: float | None = None,
        restitution_coeff: float | None = None,
    ):
        """Adds a half space plane shape to the model.

        Args:
            plane_normal: The normal direction of the plane
            plane_offset: The offset of the plane
        """
        self.shape_type.append(GeoType.HALF_SPACE_PLANE)

        if not isinstance(plane_normal, np.ndarray):
            plane_normal = np.asarray(plane_normal, dtype=np.float64)
        else:
            if plane_normal.shape != (3,):
                raise RuntimeError("plane_normal must have a shape of (3,)")
        nrm_len = np.linalg.norm(plane_normal)
        if nrm_len < 1e-8:
            raise RuntimeError(f"Plane normal ({plane_normal}) is too small")
        plane_normal = plane_normal / nrm_len

        # construct the transform matrix that transforms the plane to the world frame
        rot, _ = R.align_vectors(plane_normal, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        trans = np.eye(4)
        trans[:3, :3] = rot.as_matrix()
        trans[:3, 3] = np.array(plane_normal * plane_offset, dtype=np.float64)

        self.shape_transform.append(trans)
        self.shape_inv_transform.append(np.linalg.inv(trans))
        if stiffness is None:
            stiffness = self.DEFAULT_SHAPE_PENALTY_STIFFNESS
        elif stiffness <= 0.0:
            raise RuntimeError(f"Invalid stiffness ({stiffness} must > 0)")

        if damping is None:
            damping = self.DEFAULT_SHAPE_PENALTY_DAMPING
        elif damping < 0.0:
            raise RuntimeError(f"Invalid damping ({damping} must >= 0)")

        self.shape_penalty_params.append([stiffness, damping])
        if restitution_coeff is None:
            self.shape_restitution_coeff.append(self.DEFAULT_SHAPE_RESTITUTION_COEFF)
        else:
            self.shape_restitution_coeff.append(restitution_coeff)

    def add_gravitational(self, i: int, j: int, G: float):
        """Adds a gravitational attraction force between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            G: The gravitational constant
        """
        self.gravitational_pairs.append(i)
        self.gravitational_pairs.append(j)
        self.gravitational_constant.append(G)

    def finalize(self) -> Model:
        """
        Finalize the builder and create a concrete Model for simulation.

        This method transfers all simulation data from the builder to a Model instance,
        returning a Model object ready for simulation. It should be called after all
        elements (particles, bodies, shapes, joints, etc.) have been added to the builder.

        Returns:
            Model: A fully constructed Model object containing all simulation data.

        NOTES:
            - This method also perform necessary validation of simulation setup
        """
        m = Model()
        m.gravity = np.array(self.up_axis.to_vector(), dtype=np.float64) * self.gravity
        m.up_axis = self.up_axis

        # ---------------------
        # particles

        m.particle_count = self.particle_count
        m.particle_q = np.asarray(self.particle_q, dtype=np.float64)
        m.particle_qd = np.asarray(self.particle_qd, dtype=np.float64)
        # check all mass are positive
        for c in self.particle_mass:
            if c < 1e-8:
                raise RuntimeError(f"Particle mass ({c}) is too small")
        m.particle_mass = np.asarray(self.particle_mass, dtype=np.float64)
        m.particle_inv_mass = np.reciprocal(m.particle_mass)
        m.particle_radius = np.asarray(self.particle_radius, dtype=np.float64)
        m.particle_flags = np.asarray(self.particle_flags, dtype=np.int32)
        m.particle_drag = np.asarray(self.particle_drag, dtype=np.float64)
        m.particle_restitution_coeff = np.asarray(self.particle_restitution_coeff, dtype=np.float64)
        # For fixed particles, ensure:
        #   1) the velocity to be zero
        #   2) the inverse mass to be zero
        for i in range(self.particle_count):
            if m.particle_flags[i] & ParticleFlags.ACTIVE.value == 0:
                m.particle_qd[i] = np.zeros(3)
                m.particle_inv_mass[i] = 0.0
            if m.particle_restitution_coeff[i] < 0.0 or m.particle_restitution_coeff[i] > 1.0:
                raise RuntimeError(
                    f"Particle restitution coefficient ({m.particle_restitution_coeff[i]}) must be in [0.0, 1.0]"
                )

        # ---------------------
        # springs
        for i in self.spring_indices:
            if i < 0 or i >= self.particle_count:
                raise RuntimeError(f"Spring particle index ({i}) is out of range")
        for ke, kd in zip(self.spring_stiffness, self.spring_damping, strict=True):
            if ke < 0 or kd < 0:
                raise RuntimeError(f"Failed to satisfy (stiffness >= 0) and (damping >= 0): ke={ke}, kd={kd}")
        m.spring_indices = np.array(self.spring_indices, dtype=np.int32).reshape((-1, 2))
        m.spring_stiffness = np.array(self.spring_stiffness, dtype=np.float64)
        m.spring_damping = np.array(self.spring_damping, dtype=np.float64)
        for i in range(self.spring_count):
            r = self.spring_rest_length[i]
            if r < 0.0:
                r = np.linalg.norm(m.particle_q[m.spring_indices[i, 0]] - m.particle_q[m.spring_indices[i, 1]])
            self.spring_rest_length[i] = r
        m.spring_rest_length = np.array(self.spring_rest_length, dtype=np.float64)

        # ---------------------
        # gravitational
        for i in self.gravitational_pairs:
            if i < 0 or i >= self.particle_count:
                raise RuntimeError(f"Gravitational particle index ({i}) is out of range")
        for G in self.gravitational_constant:
            if G < 0:
                raise RuntimeError(f"Gravitational constant ({G}) is negative")
        m.gravitational_pairs = np.array(self.gravitational_pairs, dtype=np.int32).reshape((-1, 2))
        m.gravitational_constant = np.array(self.gravitational_constant, dtype=np.float64)

        # ---------------------
        # shapes
        m.shape_type = self.shape_type
        if len(self.shape_transform) > 0:
            m.shape_transform = np.stack(self.shape_transform, axis=0)
            m.shape_inv_transform = np.stack(self.shape_inv_transform, axis=0)
            m.shape_penalty_params = np.asarray(self.shape_penalty_params, dtype=np.float64)
            m.shape_restitution_coeff = np.asarray(self.shape_restitution_coeff, dtype=np.float64)
            # check all restitution coefficients are in [0.0, 1.0]
            for i in range(m.shape_count):
                if m.shape_restitution_coeff[i] < 0.0 or m.shape_restitution_coeff[i] > 1.0:
                    raise RuntimeError(
                        f"Shape restitution coefficient[{i}] ({m.shape_restitution_coeff[i]}) must be in [0.0, 1.0]"
                    )
        return m
