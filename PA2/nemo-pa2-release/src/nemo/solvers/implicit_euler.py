import numpy as np

from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_all_force_pos_jacobians, eval_all_force_vel_jacobians, eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class ImplicitEulerSolver(SolverBase):
    """Implicit Euler time integrator."""

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)
        # Maximum number of iterations for the implicit Euler solver
        # Here we use 5 as the default value
        self.maxits = 5
        # This is the error tolerance to determine when to terminate
        # the Newton iteration. If the residual f(x) is less than the
        # toleration, i.e., |f9x)| < sol, then we terminate the ieration.
        self.tol = 1e-4

        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)
        self.M = np.diag(np.repeat(np.where(mask, self.model.particle_mass, 1), 3))
        # NOTE: Feel free to add any additional initialization here
        #       to ease your implementation.

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step using Implicit Euler integrator.
        """
        if dt is None:
            dt = self.dt
        self.ts += dt
        # TODO [4]: implement linearized implicit euler here

        # Newton iteration starts here. Iterate at most self.maxits times
        # You may want to refer to the pseudo code here:
        # https://en.wikipedia.org/wiki/Newton%27s_method

        for _ in range(self.maxits):
            # A newton iteration to adjust velocity
            # - evalue force at new position
            # - solve the linear system
            # - update the velocity and position
            # check for convergence
            pass
