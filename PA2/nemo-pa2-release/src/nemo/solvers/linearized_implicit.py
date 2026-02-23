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
        N = self.model.particle_count

        # Step 1: build tentative state at q* = qⁿ + h·q̇ⁿ, q̇ = q̇ⁿ
        tmp_state = self.model.state()
        tmp_state.particle_q  = state_in.particle_q  + dt * state_in.particle_qd
        tmp_state.particle_qd = state_in.particle_qd.copy()

        # Step 2: evaluate forces at (q*, q̇ⁿ)
        tmp_state.clear_forces()
        eval_all_forces(self.model, tmp_state)
        # add gravity: F_grav[i] = mass[i] * gravity  (zero for fixed particles)
        tmp_state.particle_f += np.outer(self.masked_mass, self.model.gravity)

        # Step 3: construct linear system  A_mat · δq̇ = b
        # A_mat = M - h²·∂F/∂q - h·∂F/∂q̇
        A_mat = self.M.copy()
        eval_all_force_pos_jacobians(self.model, tmp_state, A_mat, scale=-(dt**2))
        eval_all_force_vel_jacobians(self.model, tmp_state, A_mat, scale=-dt)

        # b = h · F(q*, q̇ⁿ)
        b = dt * tmp_state.particle_f.reshape(-1)

        # Step 3.1: enforce fixed particles — zero out their rows/cols, set diagonal to 1
        for i in range(N):
            if self.model.particle_flags[i] & ParticleFlags.ACTIVE.value == 0:
                for d in range(3):
                    idx = 3 * i + d
                    A_mat[idx, :] = 0.0
                    A_mat[:, idx] = 0.0
                    A_mat[idx, idx] = 1.0
                    b[idx] = 0.0

        # Step 3.2: solve for δq̇
        delta_qd = np.linalg.solve(A_mat, b)

        # Step 4: update velocity and position
        state_out.particle_qd = state_in.particle_qd + delta_qd.reshape(N, 3)
        state_out.particle_q  = state_in.particle_q  + dt * state_out.particle_qd
