from ..core.types import nparray


class State:
    def __init__(self) -> None:
        """
        Initialize an empty State object.
        To ensure that the attributes are properly allocated create the State object via :meth:`newton.Model.state`
        instead.
        """

        self.particle_q: nparray | None = None
        """3D positions of particles, shape (particle_count, 3)"""
        self.particle_qd: nparray | None = None
        """3D velocities of particles, shape (particle_count, 3)."""
        self.particle_f: nparray | None = None
        """3D forces on particles, shape (particle_count, 3)."""

    def clear_forces(self) -> None:
        """
        Clear all force arrays (for particles and bodies) in the state object.

        Sets all entries of :attr:`particle_f` and :attr:`body_f` to zero, if present.
        """
        if self.particle_f is not None:
            self.particle_f.fill(0)

    def kinetic_energy(self, model) -> float:
        """Compute total kinetic energy: sum of 0.5 * m * ||v||^2 over active particles."""
        import numpy as np
        from ..geometry import ParticleFlags
        active = (model.particle_flags & ParticleFlags.ACTIVE.value) != 0
        v = self.particle_qd[active]
        m = model.particle_mass[active]
        return float(0.5 * np.sum(m * np.sum(v ** 2, axis=1)))
