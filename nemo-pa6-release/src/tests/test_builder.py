#
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Written by Changxi Zheng <cxz@cs.columbia.edu>, 2026
#
from pathlib import Path

import numpy as np
import pytest
import trimesh

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
    # print(m.gravity)
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


def test_builder_half_space_plane():
    builder = ModelBuilder(Axis.Y, gravity=0.2)
    builder.add_shape_half_space_plane([1.0, 0.0, 0.0], 1.0)
    v = builder.shape_inv_transform[-1] @ np.array([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0])
    v = builder.shape_inv_transform[-1] @ np.array([2.0, 0.0, 0.0, 1.0])
    assert np.allclose(v, [0.0, 1.0, 0.0, 1.0])

    builder.add_shape_half_space_plane([1.0, 1.0, 0.0], np.sqrt(2.0))
    v = builder.shape_inv_transform[-1] @ np.array([1.0, 1.0, 0.0, 1.0])
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0])

    builder.add_shape_half_space_plane([1.0, 1.0, 2.0], 1.0)
    p = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    p /= np.linalg.norm(p)
    v = builder.shape_inv_transform[-1] @ np.append(p, 1.0)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0])


def test_builder_half_space_plane2():
    builder = ModelBuilder(Axis.Y, gravity=0.2)
    builder.add_shape_half_space_plane([1.0, 0.0, 0.0], 1.0)
    builder.add_shape_half_space_plane([1.0, 1.0, 0.0], np.sqrt(2.0))
    builder.add_shape_half_space_plane([1.0, 1.0, 2.0], 1.0)
    m = builder.finalize()

    p = np.array([1.0, 1.0, 2.0], dtype=np.float64)
    p /= np.linalg.norm(p)
    v = m.shape_inv_transform[-1] @ np.append(p, 1.0)
    assert np.allclose(v, [0.0, 0.0, 0.0, 1.0])


def test_builder_half_space_plane3():
    builder = ModelBuilder()
    builder.add_shape_half_space_plane([1.0, 0.0, 1.0], np.sqrt(2.0))
    m = builder.finalize()
    p = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    v = m.shape_inv_transform[-1] @ np.append(p, 1.0)
    assert v[1] == pytest.approx(0.0)


def test_builder_shapes_0():
    builder = ModelBuilder()
    builder.add_shape_sphere([0.0, 1.0, 0.0], 1.0)
    builder.add_shape_box([0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    builder.add_shape_cylinder([0.0, 1.0, 0.0], 1.0, 1.0, [1.0, 0.0, 0.0, 0.0])
    m = builder.finalize()
    assert m.shape_count == 3
    assert m.shape_mass.shape == (3,)
    assert m.shape_inv_mass.shape == (3,)
    assert m.shape_inertia.shape == (3, 3, 3)
    assert m.shape_inv_inertia.shape == (3, 3, 3)
    assert len(m.shape_fcl_geoms) == 3
    assert m.shape_restitution_coeff.shape == (3,)


def test_builder_shapes():
    builder = ModelBuilder()
    builder.add_shape_sphere([0.0, 1.0, 0.0], 1.0)
    builder.add_shape_half_space_plane([0.0, 1.0, 0.0], 1.0)
    m = builder.finalize()
    assert m.shape_count == 2
    s = m.state()
    m.update_collision_objects(s)
    assert np.allclose(
        m.shape_collision_objects[0].getTransform().getTranslation(),
        np.asarray([0.0, 1.0, 0.0]),
    )
    assert np.allclose(
        m.shape_collision_objects[1].getTransform().getTranslation(),
        np.asarray([0.0, 0.0, 0.0]),
    )


def test_builder_mesh_box():
    mfile = Path(__file__).parents[2] / "scenes" / "meshes" / "box.obj"
    mesh = trimesh.load(mfile)
    assert mesh.volume == pytest.approx(8.0)
    assert np.allclose(mesh.moment_inertia.diagonal(), [4.0 * 2 * 8 / 12.0] * 3)
    builder = ModelBuilder()
    builder.add_shape_mesh(mesh, pos=[0.0, 1.0, 0.0], orient=[1.0, 0.0, 0.0, 0.0])
    m = builder.finalize()
    assert m.shape_count == 1
    assert m.shape_mass.shape == (1,)
    assert m.shape_inv_mass.shape == (1,)
    assert m.shape_inertia.shape == (1, 3, 3)
    assert m.shape_inv_inertia.shape == (1, 3, 3)
    assert len(m.shape_fcl_geoms) == 1
    assert m.shape_restitution_coeff.shape == (1,)
    assert np.allclose(m.shape_inertia[0].diagonal(), [4.0 * 2 * 8 / 12.0] * 3)
