import numpy as np

from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_all_force_pos_jacobians, eval_all_force_vel_jacobians, eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class LinearizedImplicitSolver(SolverBase):
    """Linearized Implicit Euler time integrator."""

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)
        # Feel free to add any additional initialization here
        # to ease your implementation.
        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        # NOTE: The masked_mass is to handle the case where some particles are fixed (see PA2 assignment notes)
        # e.g., the gravity force can be expressed as:
        #   np.outer(self.masked_mass, self.model.gravity)
        # Of course, you can use other ways to fix particles.
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)
        self.M = np.diag(np.repeat(np.where(mask, self.model.particle_mass, 1), 3))

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step using Linearized Implicit Euler integrator.
        """
        if dt is None:
            dt = self.dt
        self.ts += dt
        # TODO [4]: implement linearized implicit euler here

        # Implement your linearized implicit euler integrator algorithm here.
        # You need to perform the following steps:
        # 1. advance position only
        # 2. evalue force at new position
        # 3. solve the linear system
        # 3.1 construct the linear system
        # 3.2 solve the linear system
        # 4. update the velocity and position
