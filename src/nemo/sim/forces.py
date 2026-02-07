# from ..core.types import nparray
import numpy as np

from .model import Model
from .state import State


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
