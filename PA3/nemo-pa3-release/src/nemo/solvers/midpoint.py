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
from collections.abc import Callable

import numpy as np

from ..core.types import override
from ..geometry.types import ParticleFlags
from ..sim.forces import eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .integrator import IntegratorBase
from .solver import SolverBase


class MidpointSolver(SolverBase, IntegratorBase):
    """Midpoint time integrator.

    For now, this solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)
        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)

    @override
    def integrate(self, state_in: State, state_out: State, f: Callable[[Model, State], None], dt: float) -> None:
        """
        Integrate the model for a given time step using Symplectic Euler integrator.

        Args:
            state_in: The input state.
            state_out: The output state.
            f: The function to evaluate the force, and store the forces in the state.particle_f.
            dt: The time step.

        NOTE: This method does NOT clear the forces in the state.particle_f before calling
              the function f. This allows the caller of this method to introduce
              additional forces if needed.
        """
        f(self.model, state_in)

        h = dt * 0.5
        state_out.particle_q = state_in.particle_q + state_in.particle_qd * h
        state_out.particle_qd = state_in.particle_qd + state_in.particle_f * self.model.particle_inv_mass[:, None] * h

        state_out.clear_forces()
        f(self.model, state_out)
        state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt
        state_out.particle_qd = state_in.particle_qd + state_out.particle_f * self.model.particle_inv_mass[:, None] * dt

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

        def f(model: Model, state: State) -> None:
            eval_all_forces(model, state)
            state.particle_f += np.outer(self.masked_mass, self.model.gravity)

        self.integrate(state_in, state_out, f, dt)
