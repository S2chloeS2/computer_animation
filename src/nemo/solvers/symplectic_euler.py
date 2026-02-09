from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_spring_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class SymplecticEulerSolver(SolverBase):
    """Symplectic Euler time integrator."""

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):

        # timestep
        if dt is None:
            dt = self.dt
        self.ts += dt

        model = self.model

        # 1. clear forces
        state_in.clear_forces()

        # 2. spring forces
        eval_spring_forces(model, state_in)

        # 3. gravity force (🔥 Explicit과 동일 🔥)
        for i in range(model.particle_count):
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value:
                state_in.particle_f[i] += (
                        model.particle_mass[i] * model.gravity
                )

        # 4. Symplectic Euler integration
        for i in range(model.particle_count):

            if model.particle_flags[i] & ParticleFlags.ACTIVE.value == 0:
                state_out.particle_q[i]  = state_in.particle_q[i]
                state_out.particle_qd[i] = state_in.particle_qd[i]
                continue

            inv_m = model.particle_inv_mass[i]

            # v_{n+1}
            state_out.particle_qd[i] = (
                    state_in.particle_qd[i]
                    + dt * inv_m * state_in.particle_f[i]
            )

            # q_{n+1} = q_n + dt * v_{n+1}  (Symplectic 핵심)
            state_out.particle_q[i] = (
                    state_in.particle_q[i]
                    + dt * state_out.particle_qd[i]
            )