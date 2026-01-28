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
