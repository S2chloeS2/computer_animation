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

from nemo.sim import Model
from nemo.solvers.fluid_flip_solver import FluidFlipSolver


def test_particle_to_grid():
    model = Model()
    model.fluid_domain_res = np.asarray([2, 2], dtype=np.uint32)
    model.fluid_domain_size = np.asarray([0.2, 0.2], dtype=np.float64)
    model.fluid_cell_size = 0.1

    state = model.state()
    assert state.fluid_u_x is not None and state.fluid_u_y is not None
    assert state.fluid_u_x.shape == (2, 3)
    assert state.fluid_u_y.shape == (3, 2)

    state.fluid_particle_q = np.asarray([[0.1, 0.1]], dtype=np.float64)
    state.fluid_particle_qd = np.asarray([[0.2, 0.3]], dtype=np.float64)

    solver = FluidFlipSolver(model, dt=0.001)
    par_voxel_coord = (state.fluid_particle_q / model.fluid_cell_size).astype(np.int32)
    solver._transfer_particle_velocity_to_grid(par_voxel_coord, state)
    assert np.allclose(state.fluid_u_x[:, 1], 0.2)
    assert np.allclose(state.fluid_u_y[1, :], 0.3)


def test_particle_to_grid2():
    model = Model()
    model.fluid_domain_res = np.asarray([2, 2], dtype=np.uint32)
    model.fluid_domain_size = np.asarray([0.2, 0.2], dtype=np.float64)
    model.fluid_cell_size = 0.1

    state = model.state()
    assert state.fluid_u_x is not None and state.fluid_u_y is not None

    state.fluid_particle_q = np.asarray([[0.05, 0.05], [0.15, 0.05]], dtype=np.float64)
    state.fluid_particle_qd = np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64)

    solver = FluidFlipSolver(model, dt=0.001)
    par_voxel_coord = (state.fluid_particle_q / model.fluid_cell_size).astype(np.int32)
    solver._transfer_particle_velocity_to_grid(par_voxel_coord, state)
    assert np.allclose(state.fluid_u_x[0, 1], 0.3)
    assert np.allclose(state.fluid_u_y[1, :], [0.3, 0.5])


def test_step_static_fluid():
    model = Model()
    model.fluid_domain_res = np.asarray([2, 3], dtype=np.uint32)
    model.fluid_domain_size = np.asarray([0.2, 0.3], dtype=np.float64)
    model.fluid_cell_size = 0.1

    state = model.state()
    state.fluid_particle_q = np.asarray([[0.05, 0.05], [0.05, 0.15], [0.15, 0.05], [0.15, 0.15]], dtype=np.float64)
    state.fluid_particle_qd = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    state_out = model.state()
    state_out.fluid_particle_q = state.fluid_particle_q.copy()
    state_out.fluid_particle_qd = state.fluid_particle_qd.copy()
    solver = FluidFlipSolver(model, dt=0.001)
    solver.step(state, state_out)
    assert np.allclose(state_out.fluid_u_y, 0.0)


def test_step_static_fluid2():
    model = Model()
    N = 256
    M = 10
    model.fluid_domain_res = np.asarray([N, M], dtype=np.uint32)
    model.fluid_domain_size = np.asarray([N * 0.1, M * 0.1], dtype=np.float64)
    model.fluid_cell_size = 0.1

    state = model.state()
    state.fluid_particle_q = np.empty((7 * N, 2), dtype=np.float64)
    state.fluid_particle_qd = np.zeros((7 * N, 2), dtype=np.float64)
    for i in range(7):
        for j in range(N):
            state.fluid_particle_q[i * N + j] = [0.05 + j * 0.1, 0.05 + i * 0.1]

    state_out = model.state()
    state_out.fluid_particle_q = state.fluid_particle_q.copy()
    state_out.fluid_particle_qd = state.fluid_particle_qd.copy()
    solver = FluidFlipSolver(model, dt=0.001)
    solver.step(state, state_out)
    solver.step(state, state_out)
    # print(state_out.fluid_u_y)
    assert np.allclose(state_out.fluid_u_y[1, :], 0.0)


@pytest.mark.skip(reason="This test is just to inspect printed variables")
def test_particle_to_grid3():
    model = Model()
    model.fluid_domain_res = np.asarray([2, 2], dtype=np.uint32)
    model.fluid_domain_size = np.asarray([0.2, 0.2], dtype=np.float64)
    model.fluid_cell_size = 0.1

    state = model.state()
    assert state.fluid_u_x is not None and state.fluid_u_y is not None
    assert state.fluid_u_x.shape == (2, 3)
    assert state.fluid_u_y.shape == (3, 2)

    state.fluid_particle_q = np.asarray([[0.12, 0.1 + 0.02]], dtype=np.float64)
    state.fluid_particle_qd = np.asarray([[0.2, 0.3]], dtype=np.float64)

    solver = FluidFlipSolver(model, dt=0.001)
    par_voxel_coord = (state.fluid_particle_q / model.fluid_cell_size).astype(np.int32)
    solver._transfer_particle_velocity_to_grid(par_voxel_coord, state)
    print(state.fluid_u_x)
    print(state.fluid_u_y)
