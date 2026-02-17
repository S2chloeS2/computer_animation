from ..core.types import override
from ..geometry import ParticleFlags
from ..sim.forces import eval_spring_forces
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase
from ..sim.forces import eval_wind_forces


class MidpointSolver(SolverBase):
    """Midpoint time integrator.

    For now, this solver doesn't handle contacts.
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

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

        model = self.model

        # 1. Force at t_n
        state_in.clear_forces()
        eval_spring_forces(model, state_in)

        for i in range(model.particle_count):
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value:
                state_in.particle_f[i] += model.particle_mass[i] * model.gravity


        # 2. Build midpoint state (reuse state_out as buffer)

        for i in range(model.particle_count):

            if model.particle_flags[i] & ParticleFlags.ACTIVE.value == 0:
                state_out.particle_q[i]  = state_in.particle_q[i]
                state_out.particle_qd[i] = state_in.particle_qd[i]
                continue

            inv_m = model.particle_inv_mass[i]

            # v_{n+1/2}
            v_half = (
                    state_in.particle_qd[i]
                    + 0.5 * dt * inv_m * state_in.particle_f[i]
            )

            # x_{n+1/2}
            x_half = (
                    state_in.particle_q[i]
                    + 0.5 * dt * state_in.particle_qd[i]
            )

            state_out.particle_q[i]  = x_half
            state_out.particle_qd[i] = v_half


        # 3. Force at midpoint

        state_out.clear_forces()
        eval_spring_forces(model, state_out)

        for i in range(model.particle_count):
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value:
                state_out.particle_f[i] += model.particle_mass[i] * model.gravity


        # 3.5 wind force (bonus)
        if hasattr(self, "wind_dir") and self.wind_dir is not None:
            eval_wind_forces(
                model,
                state_out,   # midpoint state
                wind_dir=self.wind_dir,
                wind_strength=self.wind_strength,
            )

        # 4. Final update

        for i in range(model.particle_count):

            if model.particle_flags[i] & ParticleFlags.ACTIVE.value == 0:
                continue

            inv_m = model.particle_inv_mass[i]

            state_out.particle_qd[i] = (
                    state_in.particle_qd[i]
                    + dt * inv_m * state_out.particle_f[i]
            )

            state_out.particle_q[i] = (
                    state_in.particle_q[i]
                    + dt * state_out.particle_qd[i]
            )