from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_spring_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class MidpointSolver(SolverBase):
    """Midpoint time integrator.

    For now, this solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step using Midpoint integrator.

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

        # Implement your midpoint integrator algorithm here.
        # At the high-level, it will be a for loop through all particles:
        #  for i in range(self.model.particle_count):
        #    ... advance the position (in particle_q) and velocity (in particle_qd) of particle i
        #
        # The updated particle positions and velocities are stored in `state_out.particle_q` and `state_out.particle_qd`
        #
        # HINT: for midpoint method, you need to call eval_spring_forces twice. Once to evaluate force
        # at t0, another time to evalute force at x_0 + h/2 f(x_0), as discussed in the class


        # NOTE: The following code are dummy code. They are NOT a part of the implementation.
        #       They are here just to ensure the starter code can be run and the viewer can be launched.
        #       When you finish your PA, PLEASE remove the next three lines of code
        for ii in range(self.model.particle_count):
            state_out.particle_q[ii] = state_in.particle_q[ii]
            state_out.particle_qd[ii] = state_in.particle_qd[ii]

