from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_spring_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class ExplicitEulerSolver(SolverBase):
    """Explicit Euler time integrator.

    For now, this solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):

        if dt is None:
            dt = self.dt
        self.ts += dt

        model = self.model

        # 1. copy state
        state_out.particle_q[:] = state_in.particle_q
        state_out.particle_qd[:] = state_in.particle_qd

        # 2. compute forces
        state_in.clear_forces()
        eval_spring_forces(model, state_in)

        # 3. Explicit Euler
        for ii in range(model.particle_count):
            print("force:", state_in.particle_f)

            if model.particle_flags[ii] & ParticleFlags.ACTIVE.value == 0:
                continue

            inv_m = model.particle_inv_mass[ii]

            # v_{n+1} = v_n + dt * M^{-1} * F
            state_out.particle_qd[ii] = (
                    state_in.particle_qd[ii]
                    + dt * inv_m * state_in.particle_f[ii]
            )

            # q_{n+1} = q_n + dt * v_n
            state_out.particle_q[ii] = (
                    state_in.particle_q[ii]
                    + dt * state_in.particle_qd[ii]
            )


