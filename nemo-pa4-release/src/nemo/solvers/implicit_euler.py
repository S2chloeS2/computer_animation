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


class ImplicitEulerSolver(SolverBase):
    """Implicit Euler time integrator."""

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)
        # Maximum number of iterations for the implicit Euler solver
        # Here we use 5 as the default value
        self.maxits = 10
        # This is the error tolerance to determine when to terminate
        # the Newton iteration. If the residual f(x) is less than the
        # toleration, i.e., |f(x)| < sol, then we terminate the ieration.
        self.tol = 1e-4

        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)
        self.M = np.diag(np.repeat(np.where(mask, self.model.particle_mass, 1), 3))

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step using Implicit Euler integrator.
        """
        if dt is None:
            dt = self.dt
        self.ts += dt

        # Newton iteration starts here. Iterate at most self.maxits times
        # You may want to refer to the pseudo code here:
        # https://en.wikipedia.org/wiki/Newton%27s_method
        np.copyto(state_out.particle_qd, state_in.particle_qd)  # qd_0
        state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt
        for _ in range(self.maxits):
            # A newton iteration to adjust velocity
            # - evalue force at new position
            state_out.clear_forces()
            eval_all_forces(self.model, state_out)  # (q_i, qd_i) -> f_i
            state_out.particle_f += np.outer(self.masked_mass, self.model.gravity)

            # LHS
            A = self.M.copy()
            eval_all_force_pos_jacobians(self.model, state_out, A, -dt * dt)
            eval_all_force_vel_jacobians(self.model, state_out, A, -dt)
            # RHS
            b = (
                -self.M @ (state_out.particle_qd - state_in.particle_qd).reshape(-1)
                + state_out.particle_f.reshape(-1) * dt
            )
            # - solve the linear system
            delta_q = np.linalg.solve(A, b)
            # - update the velocity and position
            state_out.particle_qd += delta_q.reshape(-1, 3)  # qd_(i+1)
            state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt  # q_(i+1)

            # check for convergence
            # print(f"Iteration {iii}: {np.linalg.norm(delta_q)} {np.linalg.norm(b)}")
            if np.linalg.norm(delta_q) < self.tol:
                break
