import numpy as np

from ..core.types import Axis, nparray
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

    @property
    def spring_count(self) -> int:
        """
        The number of springs in the model.
        """
        return 0 if self.spring_rest_length is None else len(self.spring_rest_length)

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
