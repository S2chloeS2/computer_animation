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
import trimesh

from nemo.sim import ModelBuilder
from nemo.sim.sparse import ParticleIndexCache


def test_load_cloth2():
    builder = ModelBuilder()
    mesh = trimesh.load("scenes/meshes/cloth_17x17.obj")
    builder.add_cloth_mesh(mesh, pos=[0.0, 0.0, 0.0], orient=[1.0, 0.0, 1.0, 1.0], density=1.0, fixed=[0, 1])
    model = builder.finalize()
    state = model.state()

    t0, t1, t2 = model.tri_indices[4]
    aa = (
        np.column_stack((model.particle_q[t1] - model.particle_q[t0], model.particle_q[t2] - model.particle_q[t0]))
        @ model.tri_uv_inv[4]
    )

    def shear_constr(kr):
        wuv = (
            np.column_stack((state.particle_q[t1] - state.particle_q[t0], state.particle_q[t2] - state.particle_q[t0]))
            @ model.tri_uv_inv[4]
        )
        return np.dot(wuv[:, 0], wuv[:, 1])

    wu = aa[:, 0]
    wv = aa[:, 1]
    kr = model.tri_materials[4][2]
    pos = state.particle_q.copy()
    DELTA = 1e-8
    d00 = model.tri_uv_inv[4, 0, 0]
    d01 = model.tri_uv_inv[4, 0, 1]
    d10 = model.tri_uv_inv[4, 1, 0]
    d11 = model.tri_uv_inv[4, 1, 1]

    deriv = -(d00 + d10) * wv - (d01 + d11) * wu
    for i in range(3):
        state.particle_q[t0, i] = pos[t0, i] + DELTA
        e_p = shear_constr(kr)
        state.particle_q[t0, i] = pos[t0, i] - DELTA
        e_m = shear_constr(kr)
        assert (e_p - e_m) / (2 * DELTA) == pytest.approx(deriv[i])

    state.particle_q = pos.copy()
    deriv = d00 * wv + d01 * wu
    for i in range(3):
        state.particle_q[t1, i] = pos[t1, i] + DELTA
        e_p = shear_constr(kr)
        state.particle_q[t1, i] = pos[t1, i] - DELTA
        e_m = shear_constr(kr)
        assert (e_p - e_m) / (2 * DELTA) == pytest.approx(deriv[i])

    state.particle_q = pos.copy()
    deriv = d10 * wv + d11 * wu
    for i in range(3):
        state.particle_q[t2, i] = pos[t2, i] + DELTA
        e_p = shear_constr(kr)
        state.particle_q[t2, i] = pos[t2, i] - DELTA
        e_m = shear_constr(kr)
        assert (e_p - e_m) / (2 * DELTA) == pytest.approx(deriv[i])


def test_load_cloth():
    builder = ModelBuilder()
    mesh = trimesh.load("scenes/meshes/cloth_17x17.obj")
    builder.add_cloth_mesh(mesh, pos=[0.0, 0.0, 0.0], orient=[1.0, 0.0, 1.0, 1.0], density=1.0, fixed=[0, 1])
    model = builder.finalize()
    assert model.particle_count == 17 * 17

    index_cache = ParticleIndexCache(model)
    assert index_cache.N == model.particle_count
    assert index_cache.mass_mat.shape == (model.particle_count, 3, 3)
    # print(index_cache.mass_mat[2])
    # print(model.particle_mass[2])
    assert np.all(index_cache.mass_mat[3] == np.eye(3) * model.particle_mass[3])
    # print(index_cache.nonzero_count)

    t0, t1, t2 = model.tri_indices[4]
    aa = (
        np.column_stack((model.particle_q[t1] - model.particle_q[t0], model.particle_q[t2] - model.particle_q[t0]))
        @ model.tri_uv_inv[4]
    )
    assert np.allclose(aa.T @ aa, np.eye(2))

    state = model.state()
    # area = model.tri_areas[4]

    def stretch_energy(ks):
        wuv = (
            np.column_stack((state.particle_q[t1] - state.particle_q[t0], state.particle_q[t2] - state.particle_q[t0]))
            @ model.tri_uv_inv[4]
        )
        return np.linalg.norm(wuv[:, 0]) - 1.0, np.linalg.norm(wuv[:, 1]) - 1.0

    ks = model.tri_materials[4][0]
    # stretch_energy = stretch_energy(kr)
    # print(stretch_energy)

    wu = aa[:, 0] / np.linalg.norm(aa[:, 0])
    wv = aa[:, 1] / np.linalg.norm(aa[:, 1])
    pos = state.particle_q.copy()
    DELTA = 1e-8
    for i in range(3):
        state.particle_q[t0, i] = pos[t0, i] + DELTA
        eu_p, ev_p = stretch_energy(ks)
        state.particle_q[t0, i] = pos[t0, i] - DELTA
        eu_m, ev_m = stretch_energy(ks)
        assert -(model.tri_uv_inv[4, 0, 0] + model.tri_uv_inv[4, 1, 0]) * wu[i] == pytest.approx(
            (eu_p - eu_m) / (2.0 * DELTA)
        )
        assert -(model.tri_uv_inv[4, 0, 1] + model.tri_uv_inv[4, 1, 1]) * wv[i] == pytest.approx(
            (ev_p - ev_m) / (2.0 * DELTA)
        )

    state.particle_q = pos.copy()
    for i in range(3):
        state.particle_q[t2, i] = pos[t2, i] + DELTA
        eu_p, ev_p = stretch_energy(ks)
        state.particle_q[t2, i] = pos[t2, i] - DELTA
        eu_m, ev_m = stretch_energy(ks)
        assert model.tri_uv_inv[4, 1, 0] * wu[i] == pytest.approx((eu_p - eu_m) / (2.0 * DELTA))
        assert model.tri_uv_inv[4, 1, 1] * wv[i] == pytest.approx((ev_p - ev_m) / (2.0 * DELTA))

    state.particle_q = pos.copy()
    for i in range(3):
        state.particle_q[t1, i] = pos[t1, i] + DELTA
        eu_p, ev_p = stretch_energy(ks)
        state.particle_q[t1, i] = pos[t1, i] - DELTA
        eu_m, ev_m = stretch_energy(ks)
        assert model.tri_uv_inv[4, 0, 0] * wu[i] == pytest.approx((eu_p - eu_m) / (2.0 * DELTA))
        assert model.tri_uv_inv[4, 0, 1] * wv[i] == pytest.approx((ev_p - ev_m) / (2.0 * DELTA))


def test_load_cloth3():
    builder = ModelBuilder()
    mesh = trimesh.load("scenes/meshes/cloth_17x17.obj")
    builder.add_cloth_mesh(mesh, pos=[0.0, 1.0, 0.0], orient=[1.0, 1.0, 1.0, 1.0], density=1.0, fixed=[0, 1])
    model = builder.finalize()

    t0, t1, t2 = model.tri_indices[4]
    state = model.state()

    # area = model.tri_areas[4]
    def wuv():
        wuv = (
            np.column_stack((state.particle_q[t1] - state.particle_q[t0], state.particle_q[t2] - state.particle_q[t0]))
            @ model.tri_uv_inv[4]
        )
        wuv_norm = wuv.copy()
        # print("HERE", np.outer(wuv[:,0], wuv[:,0]))
        wuv_norm[:, 0] /= np.linalg.norm(wuv_norm[:, 0])
        wuv_norm[:, 1] /= np.linalg.norm(wuv_norm[:, 1])
        return wuv, wuv_norm

    state.particle_q[t1, 1] += 1e-2
    state.particle_q[t2, 0] -= 1e-2
    wuv_0, wuv_norm_0 = wuv()

    d00 = model.tri_uv_inv[4, 0, 0]
    d01 = model.tri_uv_inv[4, 0, 1]
    d10 = model.tri_uv_inv[4, 1, 0]
    d11 = model.tri_uv_inv[4, 1, 1]

    pos = state.particle_q.copy()
    DELTA = 1e-9

    deriv = -(d00 + d10) / np.linalg.norm(wuv_0[:, 0]) * (np.eye(3) - np.outer(wuv_norm_0[:, 0], wuv_norm_0[:, 0]))
    ret = []
    for i in range(3):
        state.particle_q[t0, i] = pos[t0, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t0, i] = pos[t0, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 0] - wuv_norm_m[:, 0]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)

    ret = []
    deriv = d10 / np.linalg.norm(wuv_0[:, 0]) * (np.eye(3) - np.outer(wuv_norm_0[:, 0], wuv_norm_0[:, 0]))
    state.particle_q = pos.copy()
    for i in range(3):
        state.particle_q[t2, i] = pos[t2, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t2, i] = pos[t2, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 0] - wuv_norm_m[:, 0]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)

    ret = []
    deriv = d00 / np.linalg.norm(wuv_0[:, 0]) * (np.eye(3) - np.outer(wuv_norm_0[:, 0], wuv_norm_0[:, 0]))
    state.particle_q = pos.copy()
    for i in range(3):
        state.particle_q[t1, i] = pos[t1, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t1, i] = pos[t1, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 0] - wuv_norm_m[:, 0]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)

    # -----------------------------------------------------------------
    ret = []
    deriv = -(d01 + d11) / np.linalg.norm(wuv_0[:, 1]) * (np.eye(3) - np.outer(wuv_norm_0[:, 1], wuv_norm_0[:, 1]))
    for i in range(3):
        state.particle_q[t0, i] = pos[t0, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t0, i] = pos[t0, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 1] - wuv_norm_m[:, 1]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)

    ret = []
    deriv = d01 / np.linalg.norm(wuv_0[:, 1]) * (np.eye(3) - np.outer(wuv_norm_0[:, 1], wuv_norm_0[:, 1]))
    for i in range(3):
        state.particle_q[t1, i] = pos[t1, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t1, i] = pos[t1, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 1] - wuv_norm_m[:, 1]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)

    ret = []
    deriv = d11 / np.linalg.norm(wuv_0[:, 1]) * (np.eye(3) - np.outer(wuv_norm_0[:, 1], wuv_norm_0[:, 1]))
    for i in range(3):
        state.particle_q[t2, i] = pos[t2, i] + DELTA
        _, wuv_norm_p = wuv()
        state.particle_q[t2, i] = pos[t2, i] - DELTA
        _, wuv_norm_m = wuv()
        ret.append((wuv_norm_p[:, 1] - wuv_norm_m[:, 1]) / (2 * DELTA))
    assert np.allclose(np.column_stack(ret), deriv, atol=1e-7)


def compute_bending_angle(x):
    e = x[1] - x[0]
    n1 = np.cross(e, x[2] - x[0])
    n2 = np.cross(x[3] - x[0], e)

    n1_nrm = np.linalg.norm(n1)
    n1_hat = n1 / n1_nrm
    n2_nrm = np.linalg.norm(n2)
    n2_hat = n2 / n2_nrm
    e_nrm = np.linalg.norm(e)
    e_hat = e / e_nrm

    cos_theta = np.dot(n1_hat, n2_hat)
    sin_theta = np.dot(np.cross(n1_hat, n2_hat), e_hat)
    theta = np.arctan2(sin_theta, cos_theta)

    dx2 = -n1 * (e_nrm / n1_nrm**2)
    dx3 = -n2 * (e_nrm / n2_nrm**2)

    dx0 = -dx2 * (np.dot(x[1] - x[2], e) / (e_nrm**2)) - dx3 * (np.dot(x[1] - x[3], e) / (e_nrm**2))
    dx1 = dx2 * (np.dot(x[0] - x[2], e) / (e_nrm**2)) + dx3 * (np.dot(x[0] - x[3], e) / (e_nrm**2))
    return theta, dx0, dx1, dx2, dx3


def test_bending1():
    x = np.asarray(
        [
            [0, 0, 0],  # x1
            [0, 1, 0],  # x2
            [-0.8, 0.5, -0.01],  # x3
            [0.2, 0.4, 0.7],  # x4
        ],
        dtype=np.float64,
    )
    _, dx1, dx2, dx3, dx4 = compute_bending_angle(x)

    # print(f"dx3 = {dx3}")
    # print(f"dx4 = {dx4}")
    Delta = 1e-7
    for i in range(3):
        x_plus = x.copy()
        x_plus[2, i] += Delta
        theta_plus, _, _, _, _ = compute_bending_angle(x_plus)

        x_minus = x.copy()
        x_minus[2, i] -= Delta
        theta_minus, _, _, _, _ = compute_bending_angle(x_minus)

        dtheta_dxi = (theta_plus - theta_minus) / (2 * Delta)
        # print(f"dtheta_dxi[{i}] = {dtheta_dxi}", dx3[i])
        assert np.isclose(dtheta_dxi, dx3[i])

        x_plus = x.copy()
        x_plus[3, i] += Delta
        theta_plus, _, _, _, _ = compute_bending_angle(x_plus)
        x_minus = x.copy()
        x_minus[3, i] -= Delta
        theta_minus, _, _, _, _ = compute_bending_angle(x_minus)
        dtheta_dxi = (theta_plus - theta_minus) / (2 * Delta)
        assert np.isclose(dtheta_dxi, dx4[i])

        x_plus = x.copy()
        x_plus[0, i] += Delta
        theta_plus, _, _, _, _ = compute_bending_angle(x_plus)
        x_minus = x.copy()
        x_minus[0, i] -= Delta
        theta_minus, _, _, _, _ = compute_bending_angle(x_minus)
        dtheta_dxi = (theta_plus - theta_minus) / (2 * Delta)
        # print(f"dtheta_dxi[{i}] = {dtheta_dxi}", dx1[i])
        assert np.isclose(dtheta_dxi, dx1[i])

        x_plus = x.copy()
        x_plus[1, i] += Delta
        theta_plus, _, _, _, _ = compute_bending_angle(x_plus)
        x_minus = x.copy()
        x_minus[1, i] -= Delta
        theta_minus, _, _, _, _ = compute_bending_angle(x_minus)
        dtheta_dxi = (theta_plus - theta_minus) / (2 * Delta)
        # print(f"dtheta_dxi[{i}] = {dtheta_dxi}", dx2[i])
        assert np.isclose(dtheta_dxi, dx2[i])
