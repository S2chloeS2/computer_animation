import numpy as np
from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class SymplecticEulerSolver(SolverBase):
    """Explicit Euler time integrator.

    For now, this solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step using Symplectic Euler integrator.

        Args:
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).

        NOTE:
            When dt is None, this step call will use the default timestep size
            stored in self.dt. Otherwise, the given dt will be used.
        """
        # increase simulated time
        if dt is None:
            dt = self.dt
        self.ts += dt

        eval_all_forces(self.model, state_in)

        # Implement your explicit euler algorithm here.
        # At the high-level, it will be a for loop through all particles:
        #  for i in range(self.model.particle_count):
        #    ... advance the position (in particle_q) and velocity (in particle_qd) of particle i
        #
        # The updated particle positions and velocities are stored in `state_out.particle_q` and `state_out.particle_qd`
        #
        # HINT: the inverse of mass is already computed and stored in self.model.particle_inv_mass


        # NOTE: The following code are dummy code. They are NOT a part of the implementation.
        #       They are here just to ensure the starter code can be run and the viewer can be launched.
        #       When you finish your PA, PLEASE remove the next three lines of code
        np.copyto(state_out.particle_q, state_in.particle_q)
        np.copyto(state_out.particle_qd, state_in.particle_qd)
