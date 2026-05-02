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

from ..core.types import override
from ..geometry.types import ParticleFlags
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
        # The masked_mass is to handle the case where some particles are fixed (see PA2 assignment notes)
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

        # Implement your linearized implicit euler integrator algorithm here.
        # You need to perform the following steps:
        # 1. advance position only
        np.copyto(state_out.particle_qd, state_in.particle_qd)
        state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt
        # state_out now holds (q^n + h * qd^n, qd^n)
        # 2. evalue force at new position
        state_out.clear_forces()
        eval_all_forces(self.model, state_out)
        state_out.particle_f += np.outer(self.masked_mass, self.model.gravity)
        # 3. solve the linear system
        # 3.1 construct the linear system
        A = self.M.copy()
        eval_all_force_pos_jacobians(self.model, state_out, A, -dt * dt)
        eval_all_force_vel_jacobians(self.model, state_out, A, -dt)
        # 3.2 solve the linear system
        delta_q = np.linalg.solve(A, state_out.particle_f.reshape(-1) * dt)
        # 4. update the velocity and position
        state_out.particle_qd += delta_q.reshape(-1, 3)
        state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt
