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
            # 1. linear velocity (symplectic: velocity updated first)
            state_out.shape_qd[:, :3] = (
                state_in.shape_qd[:, :3] + state_in.shape_f[:, :3] * self.model.shape_inv_mass[:, None] * dt
            )

            # 2. center of mass position (use the NEW linear velocity - symplectic!)
            active_mask = self.model.shape_flags & ShapeFlags.ACTIVE.value != 0
            state_out.shape_q[:, :3] = state_in.shape_q[:, :3].copy()
            state_out.shape_q[active_mask, :3] = (
                state_in.shape_q[active_mask, :3] + state_out.shape_qd[active_mask, :3] * dt
            )

            # 3. angular velocity and 4. orientation (quaternion) per active shape
            for i in range(self.model.shape_count):
                if not (self.model.shape_flags[i] & ShapeFlags.ACTIVE.value):
                    # keep orientation and angular velocity unchanged for static shapes
                    state_out.shape_qd[i, 3:] = state_in.shape_qd[i, 3:]
                    state_out.shape_q[i, 3:] = state_in.shape_q[i, 3:]
                    continue

                # rotation matrix from current quaternion (state_in)
                R_mat = R.from_quat(state_in.shape_q[i, 3:], scalar_first=True).as_matrix()

                # Step 1: torque from world frame → body frame
                tau_body = R_mat.T @ state_in.shape_f[i, 3:]

                # Step 2: angular velocity from world frame → body frame
                omega_body = R_mat.T @ state_in.shape_qd[i, 3:]

                # Step 3: angular acceleration in body frame (Euler's eq. with Coriolis term)
                # I_body * omega_dot = tau_body - omega_body x (I_body * omega_body)
                I_body = self.model.shape_inertia[i]
                I_body_inv = self.model.shape_inv_inertia[i]
                omega_dot_body = I_body_inv @ (tau_body - np.cross(omega_body, I_body @ omega_body))

                # Step 4: update angular velocity in body frame
                omega_body_new = omega_body + omega_dot_body * dt

                # Step 5: convert back to world frame
                omega_world_new = R_mat @ omega_body_new

                # Step 6: apply damping factor (alpha=0.999) to prevent blow-up
                omega_world_new *= 0.999

                state_out.shape_qd[i, 3:] = omega_world_new

                # Step 7: update orientation quaternion using the NEW angular velocity (symplectic!)
                # q^{n+1} = q(omega_world^{n+1} * dt) ⊗ q^n
                delta_rot = R.from_rotvec(state_out.shape_qd[i, 3:] * dt)
                q_in = R.from_quat(state_in.shape_q[i, 3:], scalar_first=True)
                state_out.shape_q[i, 3:] = (delta_rot * q_in).as_quat(scalar_first=True)

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
