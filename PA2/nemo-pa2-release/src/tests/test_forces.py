import numpy as np

from nemo.sim import ModelBuilder
from nemo.sim.forces import eval_gravitational_forces


def test_gravitational_forces():
    builder = ModelBuilder()
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_particle(pos=(1, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_gravitational(0, 1, 1.0)
    model = builder.finalize()
    state = model.state()
    eval_gravitational_forces(model, state)
    assert np.all(state.particle_f[0] == np.array([1.0, 0.0, 0.0]))
    assert np.all(state.particle_f[1] == np.array([-1.0, 0.0, 0.0]))
