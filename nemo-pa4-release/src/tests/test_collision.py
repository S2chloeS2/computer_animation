# ----------------------------------------------------------------------------
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Project code of COMS W4167 by Changxi Zheng (cxz@cs.columbia.edu)
# ----------------------------------------------------------------------------

import numpy as np
import pytest

from nemo.geometry import collision
from nemo.geometry.collision import CollisionDetector, continuous_collide_particle_particle
from nemo.sim import ModelBuilder
from nemo.sim.contacts import ContactType


def test_particle_plane_collision():
    builder = ModelBuilder()
    builder.add_shape_half_space_plane([1.0, 0.0, 0.0], 1.0)
    builder.add_particle(pos=[1.2, 1.0, 2.0], vel=[0.0, 0.0, 0.0], radius=0.5, mass=1.0)
    model = builder.finalize()
    state = model.state()
    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)

    assert len(contacts.contact_instance0) == 1
    assert contacts.contact_instance0[0] == 0
    assert contacts.contact_instance1[0] == 0
    assert contacts.contact_type[0] == ContactType.FIXED_SHAPE_PARTICLE
    assert np.allclose(contacts.contact_point0[0], [1.0, 1.0, 2.0])
    assert np.allclose(contacts.contact_point1[0], [1.0 - 0.3, 1.0, 2.0])


def test_particle_plane_collision2():
    builder = ModelBuilder()
    builder.add_shape_half_space_plane([1.0, 0.0, 1.0], np.sqrt(2.0))
    builder.add_particle(pos=[1.0, 1.0, 1.0], vel=[0.0, 0.0, 0.0], radius=0.5, mass=1.0)
    model = builder.finalize()
    state = model.state()
    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)

    assert len(contacts.contact_instance0) == 1
    assert contacts.contact_instance0[0] == 0
    assert contacts.contact_instance1[0] == 0
    assert contacts.contact_type[0] == ContactType.FIXED_SHAPE_PARTICLE
    assert np.allclose(contacts.contact_point0[0], [1.0, 1.0, 1.0])
    c = 0.5 / np.sqrt(2)
    assert np.allclose(contacts.contact_point1[0], [1.0 - c, 1.0, 1.0 - c])


def test_particle_plane_collision3():
    builder = ModelBuilder()
    builder.add_shape_half_space_plane([1.0, 2.0, 3.0], np.sqrt(1.0 + 4.0 + 9.0))
    p0 = np.asarray([1.0, 2.0, 3.0])
    dd = p0 / np.linalg.norm(p0)
    builder.add_particle(pos=p0 + dd * 0.2, vel=[0.0, 0.0, 0.0], radius=0.5, mass=1.0)
    model = builder.finalize()
    state = model.state()
    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)

    assert len(contacts.contact_instance0) == 1
    assert contacts.contact_instance0[0] == 0
    assert contacts.contact_instance1[0] == 0
    assert contacts.contact_type[0] == ContactType.FIXED_SHAPE_PARTICLE
    d = np.linalg.norm(contacts.contact_point0[0] - contacts.contact_point1[0])
    assert d == pytest.approx(0.3)


def test_particle_particle_collision():
    pos0 = np.asarray([0.0, 0.0, 0.0])
    pos1 = np.asarray([1.0, 1.0, 1.0])
    depth, c0, c1 = collision.collide_particle_particle(pos0, pos1, 1.0, 1.0)
    assert depth == pytest.approx(np.sqrt(3) - 2.0)
    assert np.allclose(c0, np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3))
    assert np.allclose(c1, np.asarray([1.0, 1.0, 1.0]) - np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3))

    pos1 = np.asarray([1.0, 1.0, 0.0])
    depth, c0, _ = collision.collide_particle_particle(pos0, pos1, 1.0, 1.0)
    assert depth == pytest.approx(np.sqrt(2) - 2.0)
    assert np.allclose(c0, np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2))


def test_particle_particle_continuous_collision():
    pos0 = np.asarray([0.0, 0.0, 0.0])
    pos1 = np.asarray([1.0, 1.0, 1.0])
    rad0 = np.sqrt(3) * 0.25
    rad1 = np.sqrt(3) * 0.75
    pos0_in = np.asarray([0.0, 3.0, -1.0])
    pos1_in = np.asarray([1.0, -1.0, 2.0])

    pos0_out = (pos0 - pos0_in) * 1.5 + pos0_in
    pos1_out = (pos1 - pos1_in) * 1.5 + pos1_in
    t, n = continuous_collide_particle_particle(pos0_in, pos0_out, pos1_in, pos1_out, rad0, rad1)

    p0 = pos0_in + t * (pos0_out - pos0_in)
    p1 = pos1_in + t * (pos1_out - pos1_in)
    assert np.linalg.norm(p0 - p1) == pytest.approx(rad0 + rad1)
    assert np.allclose(n, (p1 - p0) / np.linalg.norm(p1 - p0))


def test_particle_particle_continuous_collision2():
    pos0 = np.asarray([0.0, 0.0, 0.0])
    pos1 = np.asarray([1.0, 1.0, 1.0])
    rad0 = np.sqrt(3) * 0.75
    rad1 = np.sqrt(3) * 0.75
    pos0_in = np.asarray([0.0, 3.0, -1.0])
    pos1_in = np.asarray([1.0, -1.0, 2.0])

    pos0_out = (pos0 - pos0_in) * 1.5 + pos0_in
    pos1_out = (pos1 - pos1_in) * 1.5 + pos1_in
    ret = continuous_collide_particle_particle(pos0, pos0_out, pos1, pos1_out, rad0, rad1)
    assert ret is None

    v0 = pos0_out - pos0
    v1 = pos1_out - pos1
    assert np.dot(v1 - v0, pos1 - pos0) >= 0.0


def test_particle_particle_continuous_collision3():
    pos0 = np.asarray([0.0, 0.0, 0.0])
    pos1 = np.asarray([0.0, 1.0, 1.0])
    pos1_m = np.asarray([0.0, -1.0, 1.0])
    rad = 0.5
    ret = continuous_collide_particle_particle(pos0, pos0, pos1, pos1_m, rad, rad)
    assert ret[0] == pytest.approx(0.5)


def test_shape_shape_collision():
    builder = ModelBuilder()
    builder.add_shape_sphere([0.0, 1.0, 0.0], 1.0)
    builder.add_shape_half_space_plane([1.0, 0.0, 0.0], -1.2)
    model = builder.finalize()
    state = model.state()

    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)
    assert len(contacts.contact_type) == 0

    # move the sphere
    state.shape_q[0, :3] = np.asarray([-0.3, 1.0, 2.0])
    contacts = cd.instantaneous_contacts(state)
    assert len(contacts.contact_type) == 1
    assert contacts.contact_depth[0] == pytest.approx(0.1)
    assert np.allclose(contacts.contact_normal[0], np.asarray([1.0, 0.0, 0.0]))

    state.shape_q[0, 3:] = np.asarray([np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)])
    contacts = cd.instantaneous_contacts(state)
    assert len(contacts.contact_type) == 1
    assert contacts.contact_depth[0] == pytest.approx(0.1)
    assert np.allclose(contacts.contact_normal[0], np.asarray([1.0, 0.0, 0.0]))


def test_shape_shape_collision_box():
    builder = ModelBuilder()
    h = np.sqrt(2) - 0.01
    builder.add_shape_box(pos=[1.0, h, -1.0], size=[3.0, 2.0, 2.0], orient=[1.0, 0.0, 0.0, 0.0])
    builder.add_shape_half_space_plane([0.0, 1.0, 0.0], 0.0)
    model = builder.finalize()
    state = model.state()

    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)
    # print(contacts.contact_point0)
    assert len(contacts.contact_type) == 0

    # Now rotate the box by 45 degrees
    state.shape_q[0, 3:] = np.asarray([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
    contacts = cd.instantaneous_contacts(state)
    # print(contacts.contact_point0)
    assert len(contacts.contact_type) == 1
    assert contacts.contact_depth[0] == pytest.approx(0.01)


def test_shape_shape_collision_cylinder():
    builder = ModelBuilder()
    h = np.sqrt(2) - 0.01
    builder.add_shape_cylinder(pos=[0.0, h, 0.0], radius=1.0, lz=2.0, orient=[1.0, 0.0, 0.0, 0.0])
    builder.add_shape_half_space_plane([0.0, 1.0, 0.0], 0.0)
    model = builder.finalize()
    state = model.state()
    cd = CollisionDetector(model)
    contacts = cd.instantaneous_contacts(state)
    assert len(contacts.contact_type) == 0

    # Now rotate the box by 45 degrees
    state.shape_q[0, 3:] = np.asarray([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
    contacts = cd.instantaneous_contacts(state)
    # print(contacts.contact_point0)
    assert len(contacts.contact_type) == 1
    assert contacts.contact_depth[0] == pytest.approx(0.01)
    assert np.allclose(contacts.contact_point0[0], np.asarray([0.0, -0.005, 0.0]))


def test_shape_shape_collision_cylinder2():
    builder = ModelBuilder()
    builder.add_shape_cylinder(pos=[0.0, 0.0, 0.0], radius=1.0, lz=4.0, orient=[1.0, 0.0, 0.0, 0.0])
    h = 3.0 / np.sqrt(2)
    builder.add_shape_half_space_plane([0.0, 1.0, 0.0], -(h - 0.0001))
    model = builder.finalize()
    state = model.state()

    cd = CollisionDetector(model)
    state.shape_q[0, 3:] = np.asarray([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])
    contacts = cd.instantaneous_contacts(state)
    assert len(contacts.contact_type) == 1
    assert contacts.contact_point0[0][2] == pytest.approx(1 / np.sqrt(2))
