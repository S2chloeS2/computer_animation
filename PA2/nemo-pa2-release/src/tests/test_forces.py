import numpy as np

from nemo.sim import ModelBuilder
from nemo.sim.forces import eval_drag_forces, eval_gravitational_forces, eval_spring_forces


def test_spring_forces_basic():
    builder = ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(2, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_spring(0, 1, ke=10.0, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    eval_spring_forces(model, state)
    # spring stretched by 1 unit -> force = ke * (l - l0) = 10.0 along +x on particle 1
    assert np.allclose(state.particle_f[1], np.array([-10.0, 0.0, 0.0]))


def test_drag_forces_basic():
    builder = ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(0, 0, 0), vel=(2.0, 0, 0), mass=1.0, drag=0.5)
    model = builder.finalize()
    state = model.state()
    eval_drag_forces(model, state)
    assert np.allclose(state.particle_f[0], np.array([-1.0, 0.0, 0.0]))


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


def test_spring_forces_with_damping():
    builder = ModelBuilder(gravity=0.0)
    # particle 0 fixed, particle 1 moving away at vel 1 m/s
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(1, 0, 0), vel=(1.0, 0, 0), mass=1.0)
    builder.add_spring(0, 1, ke=0.0, kd=2.0, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    eval_spring_forces(model, state)
    # only damping: F_d = kd * (v_rel · n̂) = 2.0 * 1.0 = 2.0, opposes motion -> -x
    assert np.allclose(state.particle_f[1], np.array([-2.0, 0.0, 0.0]))
