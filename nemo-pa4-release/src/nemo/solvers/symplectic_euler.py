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
from scipy.spatial.transform import Rotation as R

from ..core.types import override
from ..geometry.types import ParticleFlags, ShapeFlags
from ..sim.forces import eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .integrator import IntegratorBase
from .solver import SolverBase


class SymplecticEulerSolver(SolverBase, IntegratorBase):
    """Explicit Euler time integrator.

    NOTE: This solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

        if self.model.particle_count > 0:
            mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
            self.masked_particle_mass = np.where(mask, self.model.particle_mass, 0.0)

        if self.model.shape_count > 0:
            mask = self.model.shape_flags & ShapeFlags.ACTIVE.value != 0
            self.masked_shape_mass = np.where(mask, self.model.shape_mass, 0.0)

    @override
    def integrate(self, state_in: State, state_out: State, f: Callable[[Model, State], None], dt: float) -> None:
        """
        Integrate the model for a given time step using Symplectic Euler integrator.

        Args:
            state_in: The input state.
            state_out: The output state.
            f: The function to evaluate the force, and store the forces in state.particle_f and state.shape_f.
            dt: The time step.

        NOTE: This method does NOT clear the forces in the state.particle_f before calling
              the function f. This allows the caller of this method to introduce
              additional forces if needed.
        """
        # particle_f and shape_f are stored in state_in
        f(self.model, state_in)
        if self.model.particle_count > 0:
            state_out.particle_qd = (
                state_in.particle_qd + state_in.particle_f * self.model.particle_inv_mass[:, None] * dt
            )
            state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt
        # integrate rigid body states
        if self.model.shape_count > 0:
            # 1. linear velocity
            state_out.shape_qd[:, :3] = (
                state_in.shape_qd[:, :3] + state_in.shape_f[:, :3] * self.model.shape_inv_mass[:, None] * dt
            )
            # TODO: Finish the integration of the rigid body states.
            # I already provided the code to update linear velocity above.
            # You need to update 1) the center of mass position, 2) the angular velocity and 3) the orientation of 
            # the rigid body.
            #
            # Please refer to Sec. 1.1.5 of the course notes.

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

        def f(model: Model, state: State) -> None:
            state.clear_forces()
            eval_all_forces(model, state)
            if self.model.particle_count > 0:
                state.particle_f += np.outer(self.masked_particle_mass, self.model.gravity)
            if self.model.shape_count > 0:
                state.shape_f[:, :3] += np.outer(self.masked_shape_mass, self.model.gravity)

        self.integrate(state_in, state_out, f, dt)
