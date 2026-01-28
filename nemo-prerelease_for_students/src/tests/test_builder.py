import numpy as np
import pytest

from nemo.core import Axis
from nemo.sim import ModelBuilder


def test_builder_particles():
    builder = ModelBuilder()
    builder.add_particle(
        pos=(1, 1, 1),
        vel=(0, 0, 0),
        mass=1.0,
    )
    m = builder.finalize()
    assert m.particle_count == 1


def test_builder_particles_fail():
    builder = ModelBuilder()
    builder.add_particle(
        pos=(1, 1, 1),
        vel=(0, 0, 0),
        mass=0.0,
    )
    with pytest.raises(RuntimeError):
        # expect an error due to zero mass
        builder.finalize()


def test_builder_particles2():
    builder = ModelBuilder()
    builder.add_particle(
        pos=(1, 1, 1),
        vel=(0, 0, 0),
        mass=1.0,
    )
    builder.add_particle(
        pos=(1, 2, 1),
        vel=(0, 0, 0),
        mass=2.0,
    )
    m = builder.finalize()
    assert m.particle_count == 2
    s = m.state()
    assert s.particle_f.shape == (2, 3)


def test_builder_gravity():
    builder = ModelBuilder(Axis.Y, gravity=0.2)
    builder.add_particle(
        pos=(1, 1, 1),
        vel=(0, 0, 0),
        mass=1.0,
    )
    m = builder.finalize()
    print(m.gravity)
    assert np.array_equal(m.gravity, (0.0, 0.2, 0.0))


def test_builder_springs():
    builder = ModelBuilder(Axis.Y, gravity=0.2)
    builder.add_particle(
        pos=(1, 1, 1),
        vel=(0, 0, 0),
        mass=1.0,
    )
    builder.add_particle(
        pos=(1, 2, 1),
        vel=(0, 0, 0),
        mass=2.0,
    )
    builder.add_particle(
        pos=(1, 2, 1),
        vel=(0, 0, 0),
        mass=1.0,
    )
    builder.add_spring(0, 1, 10.0)
    builder.add_spring(1, 2, 10.0)
    builder.add_spring(0, 2, 10.0)
    builder.add_spring(0, 2, 10.0, rest_length=1.2)
    m = builder.finalize()
    assert m.spring_indices[0, 0] == 0
    assert m.spring_indices[0, 1] == 1
    assert m.spring_indices[1, 1] == 2
    assert m.spring_indices[2, 0] == 0
    assert m.spring_indices[2, 1] == 2
    assert m.spring_rest_length[0] == 1.0
    assert m.spring_rest_length[1] == 0.0
    assert m.spring_rest_length[3] == 1.2
