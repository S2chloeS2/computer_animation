from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_spring_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase
from ..sim.forces import eval_wind_forces



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

        # 1. clear forces
        state_in.clear_forces()

        # 2. spring forces
        eval_spring_forces(model, state_in)


        # 3.5 wind force (bonus, conditional)
        if hasattr(self, "wind_dir") and self.wind_dir is not None:
            eval_wind_forces(
                model,
                state_in,
                wind_dir=self.wind_dir,
                wind_strength=self.wind_strength,
            )
        # 3. gravity (과제 요구사항)
        for i in range(model.particle_count):
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value:
                state_in.particle_f[i] += model.particle_mass[i] * model.gravity


        # 4. Explicit Euler integration
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

            # q_{n+1} = q_n + dt * v_n  (Explicit!)
            state_out.particle_q[i] = (
                    state_in.particle_q[i]
                    + dt * state_in.particle_qd[i]
            )

