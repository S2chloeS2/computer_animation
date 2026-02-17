# from ..core.types import nparray
import numpy as np

from .model import Model
from .state import State
from ..geometry import ParticleFlags



def eval_spring_forces(model: Model, state: State) -> None:
    """
    Evaluate the spring forces of the given model, and store the forces
    in `state.particle_f`
    """
    for s in range(model.spring_count):
        i, j = model.spring_indices[s]

        # relative dir of i w.r.t. j;  vec(j-->i)
        dir = state.particle_q[i] - state.particle_q[j]
        nrm = np.linalg.norm(dir)  # distance
        # damping force: d * v
        # relative vel of i w.r.t. j
        f_d = (state.particle_qd[i] - state.particle_qd[j]) * model.spring_damping[s]
        # spring force
        if nrm > 1e-10:
            f_s = dir * ((model.spring_rest_length[s] - nrm) * model.spring_stiffness[s] / nrm)
            f_tot = f_s - f_d
        else:
            f_tot = -f_d

        state.particle_f[i] += f_tot
        state.particle_f[j] -= f_tot

def eval_wind_forces(
        model: Model,
        state: State,
        wind_dir: np.ndarray = np.array([1.0, 0.0, 0.0]),
        wind_strength: float = 2.0,
) -> None:
    """
    Apply a constant wind force to all active particles.
    """

    norm = np.linalg.norm(wind_dir)
    if norm < 1e-8:
        return
    wind_dir = wind_dir / norm

    wind_force = wind_strength * wind_dir

    for i in range(model.particle_count):
        if model.particle_flags[i] & ParticleFlags.ACTIVE.value:
            state.particle_f[i] += wind_force