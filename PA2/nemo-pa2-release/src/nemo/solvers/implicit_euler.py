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
        N = self.model.particle_count

        # precompute fixed-DOF indices once
        fixed_dofs = [
            3 * i + d
            for i in range(N)
            if self.model.particle_flags[i] & ParticleFlags.ACTIVE.value == 0
            for d in range(3)
        ]

        # initial guess: v₀ = q̇ⁿ
        v = state_in.particle_qd.copy()   # shape (N, 3)

        # Newton iteration starts here. Iterate at most self.maxits times
        for _ in range(self.maxits):
            # q* = qⁿ + h·vᵢ
            tmp_state = self.model.state()
            tmp_state.particle_q  = state_in.particle_q + dt * v
            tmp_state.particle_qd = v.copy()

            # evaluate F(q*, vᵢ) including gravity
            tmp_state.clear_forces()
            eval_all_forces(self.model, tmp_state)
            tmp_state.particle_f += np.outer(self.masked_mass, self.model.gravity)

            # build Jacobian of R:  A_mat = M - h²·∂F/∂q - h·∂F/∂q̇
            A_mat = self.M.copy()
            eval_all_force_pos_jacobians(self.model, tmp_state, A_mat, scale=-(dt**2))
            eval_all_force_vel_jacobians(self.model, tmp_state, A_mat, scale=-dt)

            # rhs = -R(vᵢ) = -M(vᵢ - q̇ⁿ) + h·F
            Mv_diff = self.M @ (v - state_in.particle_qd).reshape(-1)
            b = -Mv_diff + dt * tmp_state.particle_f.reshape(-1)

            # enforce fixed particles
            for idx in fixed_dofs:
                A_mat[idx, :] = 0.0
                A_mat[:, idx] = 0.0
                A_mat[idx, idx] = 1.0
                b[idx] = 0.0

            # solve for δv and update
            delta_v = np.linalg.solve(A_mat, b)
            v = v + delta_v.reshape(N, 3)

            # check for convergence: ‖δv‖ < tol
            if np.linalg.norm(delta_v) < self.tol:
                break

        # write final state
        state_out.particle_qd = v
        state_out.particle_q  = state_in.particle_q + dt * v
