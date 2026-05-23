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
import numpy as np

from nemo.sim.sparse import SparseParticleSystem

from ..core.types import nparray
from ..geometry.types import ParticleFlags
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
        if nrm > 1e-10:
            # damping force: d * v
            # relative vel of i w.r.t. j
            dir /= nrm  # normalize the direction
            f_s = dir * ((model.spring_rest_length[s] - nrm) * model.spring_stiffness[s])
            if model.spring_damping[s] > 0:
                f_d = dir * np.dot(state.particle_qd[i] - state.particle_qd[j], dir) * model.spring_damping[s]
                f_tot = f_s - f_d
            else:
                f_tot = f_s
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                state.particle_f[i] += f_tot
            if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                state.particle_f[j] -= f_tot


def eval_spring_force_pos_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Evaluate the spring force jacobians with respect to the position,
    and accumulate the jacobians into the given array A.

    This is to compute A = A + s * (partial F / partial q)

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        s: float: the scalar to scale the Jacobian before adding to A
    """
    for s in range(model.spring_count):
        i, j = model.spring_indices[s]
        n = state.particle_q[i] - state.particle_q[j]
        ld = np.linalg.norm(n)
        if ld > 1e-10:
            n /= ld
            outer_n = np.outer(n, n)
            i3 = i * 3
            j3 = j * 3
            # ------ spring force jacobian ------
            Ks = (outer_n + (np.eye(3) - outer_n) * (ld - model.spring_rest_length[s]) / ld) * model.spring_stiffness[s]
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                A[i3 : i3 + 3, i3 : i3 + 3] -= scale * Ks
                if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                    A[i3 : i3 + 3, j3 : j3 + 3] += scale * Ks

            if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                A[j3 : j3 + 3, j3 : j3 + 3] -= scale * Ks
                if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                    A[j3 : j3 + 3, i3 : i3 + 3] += scale * Ks

            # ------ damping force jacobian ------
            beta = model.spring_damping[s]
            if beta > 0:
                vd = state.particle_qd[i] - state.particle_qd[j]
                Kd = ((np.dot(n, vd) * np.eye(3) + np.outer(n, vd)) @ (np.eye(3) - outer_n)) * (beta / ld)
                if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                    A[i3 : i3 + 3, i3 : i3 + 3] -= scale * Kd
                    if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                        A[i3 : i3 + 3, j3 : j3 + 3] += scale * Kd
                if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                    A[j3 : j3 + 3, j3 : j3 + 3] -= scale * Kd
                    if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                        A[j3 : j3 + 3, i3 : i3 + 3] += scale * Kd


def eval_spring_force_vel_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Evaluate the spring force jacobians with respect to the velocity,
    and accumulate the jacobians into the given array A.

    This is to compute A = A + scale * (partial F / partial dot[q])

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        scale: float: the scalar to scale the Jacobian before adding to A
    """
    for s in range(model.spring_count):
        i, j = model.spring_indices[s]
        n = state.particle_q[i] - state.particle_q[j]
        ld = np.linalg.norm(n)
        beta = model.spring_damping[s]
        if ld > 1e-10 and beta > 0:
            n /= ld
            outer_n = np.outer(n, n)
            i3 = i * 3
            j3 = j * 3
            sb = scale * beta
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                A[i3 : i3 + 3, i3 : i3 + 3] -= outer_n * sb
                if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                    A[i3 : i3 + 3, j3 : j3 + 3] += outer_n * sb
            if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                A[j3 : j3 + 3, j3 : j3 + 3] -= outer_n * sb
                if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                    A[j3 : j3 + 3, i3 : i3 + 3] += outer_n * sb


def eval_gravitational_forces(model: Model, state: State) -> None:
    """
    Evaluate the gravitational forces of the given model, and store the forces
    in `state.particle_f`

    NOTE: This function does not consider the gravity of the model.
    It only considers the gravitational force between two particles.
    The gravity of the model is considered separately.
    """
    for g in range(model.gravitational_count):
        i, j = model.gravitational_pairs[g]
        dir = state.particle_q[i] - state.particle_q[j]  # vector from j to i
        nrm = np.linalg.norm(dir)
        if nrm > 1e-10:
            f_g = dir * (model.gravitational_constant[g] * model.particle_mass[i] * model.particle_mass[j] / nrm**3)
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                state.particle_f[i] -= f_g
            if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                state.particle_f[j] += f_g


def eval_gravitational_force_pos_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Evaluate the gravitational force jacobians with respect to the position,
    and store the jacobians into the given array A.

    This is to compute A = A + scale * (partial F / partial q)

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        scale: float: the scalar to scale the Jacobian before adding to A
    """
    for g in range(model.gravitational_count):
        i, j = model.gravitational_pairs[g]
        dir = state.particle_q[i] - state.particle_q[j]  # vector from j to i
        nrm = np.linalg.norm(dir)
        # when nrm is positive (i.e., nrm > 1e-10), compute K
        # and then add K (and -K) to the corresponding rows and columns of A
        if nrm > 1e-10:
            G = model.gravitational_constant[g]
            m1 = model.particle_mass[i]
            m2 = model.particle_mass[j]
            i3 = i * 3
            j3 = j * 3
            Ks = (-G * m1 * m2 / nrm**3) * (np.eye(3) - np.outer(dir, dir) * (3.0 / nrm**2))
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                A[i3 : i3 + 3, i3 : i3 + 3] += scale * Ks
                if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                    A[i3 : i3 + 3, j3 : j3 + 3] -= scale * Ks

            if model.particle_flags[j] & ParticleFlags.ACTIVE.value != 0:
                A[j3 : j3 + 3, j3 : j3 + 3] += scale * Ks
                if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
                    A[j3 : j3 + 3, i3 : i3 + 3] -= scale * Ks


def eval_drag_forces(model: Model, state: State) -> None:
    """
    Evaluate the drag forces of the given model, and store the forces
    in `state.particle_f`
    """
    for i in range(model.particle_count):
        beta = model.particle_drag[i]
        if beta > 0 and model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
            state.particle_f[i] -= state.particle_qd[i] * beta


def eval_drag_force_vel_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Evaluate the drag force jacobians with respect to the velocity of the given model,
    and store the jacobians into the given array A.

    This is to compute A = A + s * (partial F / partial dot[q])

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        s: float: the scalar to scale the Jacobian before adding to A
    """
    for i in range(model.particle_count):
        beta = model.particle_drag[i]
        if beta > 0 and model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:
            A[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] -= np.eye(3) * (scale * beta)


def eval_all_forces(model: Model, state: State) -> None:
    """
    Evaluate all the forces of the given model, and store the forces
    in `state.particle_f`

    NOTE: This function does not consider the gravity force (i.e., m*g).
          You need to add the gravity force to the force vector outside of this function.
    """
    eval_spring_forces(model, state)
    eval_gravitational_forces(model, state)
    eval_drag_forces(model, state)


def eval_all_force_pos_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Eval all force jacobians of the given model, and store the jacobians
    into the given array A.

    This is to compute A = A + scale * (partial F / partial q)

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        scale: float: the scalar to scale the Jacobian before adding to A
    """
    eval_spring_force_pos_jacobians(model, state, A, scale=scale)
    eval_gravitational_force_pos_jacobians(model, state, A, scale=scale)


def eval_all_force_vel_jacobians(model: Model, state: State, A: nparray, scale: float = 1.0) -> None:
    """
    Eval all force jacobians of the given model, and store the jacobians
    into the given array A.

    This is to compute A = A + scale * (partial F / partial dot[q])

    Args:
        model: Model
        state: State
        A: nparray, shape (particle_countx3, particle_countx3): output array for the jacobians
        scale: float: the scalar to scale the Jacobian before adding to A
    """
    eval_spring_force_vel_jacobians(model, state, A, scale=scale)
    eval_drag_force_vel_jacobians(model, state, A, scale=scale)


def cloth_wuv(model: Model, state: State, tri: int) -> nparray:
    """
    Return a 3x2 matrix [w_u w_v] = [x1-x0 x2-x0] @ [d00 d01; d10 d11]
    This is the deformation gradient matrix for cloth triangle (t0, t1, t2)
    """
    t0, t1, t2 = model.tri_indices[tri]
    return (
        np.column_stack(
            (state.particle_q[t1] - state.particle_q[t0], state.particle_q[t2] - state.particle_q[t0]),
        )
        @ model.tri_uv_inv[tri]
    )


def eval_cloth_stretch_shear_forces(model: Model, state: State, A: SparseParticleSystem, h: float) -> None:
    """
    This function does two things:
    1. Evaluate the stretch and shear forces of the given model, and accumulate the forces
    in `state.particle_f`,
    2. Accumulate the force Jacobians into the SparseParticleSystem A, which is the LHS of the
    linearized implicit Euler system.

    The stretch and shear forces are defined in Sec. 4.2 in the paper "Large Steps in Cloth Simulation"

    When returning, `A` stores the LHS of the linearized implicit system (see PA5 course notes)

    Args:
        model: Model
        state: State
        A: SparseParticleSystem: output array for the jacobians
        h: float: the timestep size
    """
    h2 = h * h
    for tri in range(model.tri_count):
        t0, t1, t2 = model.tri_indices[tri]
        ks, ksd, kr, krd = model.tri_materials[tri]
        # Here is how to compute the cloth deformation gradient (3x2 matrix)
        wuv = cloth_wuv(model, state, tri)

        # Hint: For fixed vertices, make sure the rows in matrix A corresponding to the fixed vertex
        # are all zeros except the diagonal entry is a 3x3 diagonal mass matrix.
        # Also, make sure the right-hand side (stored in state.particle_f) is zero for fixed vertices.
        # In this way, the `delta_qd` resulted from solving the linear system in Line 77 of
        # `large_step_cloth.py` will be zero for fixed vertices.
        #
        # For example,
        # when compute the forces, check if the vertex is fixed, i.e.,
        #
        # if model.particle_flags[t0] & ParticleFlags.ACTIVE.value != 0:
        #     ...
        # Only add forces to the vertices that are active.

        # Hint: according to the derivation in course notes, the matrix A must be symmetric.
        #
        # Hint: use A.accumu_block(...) to add a 3x3 block into A. For example, each
        # (\partial C/\partial x_i)(\partial C/\partial x_j)^T should be added to
        # the [i, j] block of A, if both i and j are active vertices.
        # If either i or j is a fixed vertex, the block should be zero (think why)

        a = model.tri_areas[tri]
        d = model.tri_uv_inv[tri]  # shape (2,2): [[d00,d01],[d10,d11]]
        w_u = wuv[:, 0]
        w_v = wuv[:, 1]

        # Scalar d-coefficients: vertex i contributes d_u[i]*I to ∂w_u/∂x_i
        d_u = [-(d[0, 0] + d[1, 0]), d[0, 0], d[1, 0]]
        d_v = [-(d[0, 1] + d[1, 1]), d[0, 1], d[1, 1]]
        I3d = np.eye(3)

        # ----------------------------------------------------------------
        # Shear Force (eq 1.10–1.12, no damping yet)
        # C = w_u^T * w_v   (no area)
        # f_i = -a * kr * C * ∇C_i
        # ----------------------------------------------------------------
        C_shear = np.dot(w_u, w_v)

        # ∇C per vertex (eq 1.11)
        gs0 = -(d[0, 0] + d[1, 0]) * w_v - (d[0, 1] + d[1, 1]) * w_u
        gs1 =   d[0, 0] * w_v + d[0, 1] * w_u
        gs2 =   d[1, 0] * w_v + d[1, 1] * w_u

        t0_active = model.particle_flags[t0] & ParticleFlags.ACTIVE.value != 0
        t1_active = model.particle_flags[t1] & ParticleFlags.ACTIVE.value != 0
        t2_active = model.particle_flags[t2] & ParticleFlags.ACTIVE.value != 0

        # Shear forces
        shear_scale = a * kr * C_shear
        if t0_active:
            state.particle_f[t0] -= shear_scale * gs0
        if t1_active:
            state.particle_f[t1] -= shear_scale * gs1
        if t2_active:
            state.particle_f[t2] -= shear_scale * gs2

        # Shear Jacobian: +h² * a * kr * outer(gs_i, gs_j)  for i≤j, both active
        h2akr = h2 * a * kr
        verts = [(t0, gs0, t0_active), (t1, gs1, t1_active), (t2, gs2, t2_active)]
        for idx_i, (ti, gi, ai) in enumerate(verts):
            for tj, gj, aj in verts[idx_i:]:
                if ai and aj:
                    i_idx, j_idx = (ti, tj) if ti <= tj else (tj, ti)
                    gi_ord, gj_ord = (gi, gj) if ti <= tj else (gj, gi)
                    A.accumu_block(i_idx, j_idx, h2akr * np.outer(gi_ord, gj_ord))

        # ----------------------------------------------------------------
        # Stretch Force (eq 1.5–1.9, no damping yet)
        # C_u = ||w_u|| - 1,  C_v = ||w_v|| - 1   (no area)
        # f_i = -a * ks * (C_u * ∇Cu_i + C_v * ∇Cv_i)
        # ----------------------------------------------------------------
        wu_n = np.linalg.norm(w_u)
        wv_n = np.linalg.norm(w_v)

        if wu_n > 1e-10 and wv_n > 1e-10:
            wu_hat = w_u / wu_n
            wv_hat = w_v / wv_n
            C_u = wu_n - 1.0
            C_v = wv_n - 1.0

            # ∇C_u per vertex (eq 1.8)
            gu0 = -(d[0, 0] + d[1, 0]) * wu_hat
            gu1 =   d[0, 0] * wu_hat
            gu2 =   d[1, 0] * wu_hat
            # ∇C_v per vertex (eq 1.8)
            gv0 = -(d[0, 1] + d[1, 1]) * wv_hat
            gv1 =   d[0, 1] * wv_hat
            gv2 =   d[1, 1] * wv_hat

            # Stretch forces
            aks = a * ks
            if t0_active:
                state.particle_f[t0] -= aks * (C_u * gu0 + C_v * gv0)
            if t1_active:
                state.particle_f[t1] -= aks * (C_u * gu1 + C_v * gv1)
            if t2_active:
                state.particle_f[t2] -= aks * (C_u * gu2 + C_v * gv2)

            # Stretch Jacobian: +h² * a * ks * (outer(gu_i,gu_j) + outer(gv_i,gv_j))
            h2aks = h2 * aks
            stretch_verts = [
                (t0, gu0, gv0, t0_active),
                (t1, gu1, gv1, t1_active),
                (t2, gu2, gv2, t2_active),
            ]
            for idx_i, (ti, gui, gvi, ai) in enumerate(stretch_verts):
                for tj, guj, gvj, aj in stretch_verts[idx_i:]:
                    if ai and aj:
                        i_idx, j_idx = (ti, tj) if ti <= tj else (tj, ti)
                        if ti <= tj:
                            blk = np.outer(gui, guj) + np.outer(gvi, gvj)
                        else:
                            blk = np.outer(guj, gui) + np.outer(gvj, gvi)
                        A.accumu_block(i_idx, j_idx, h2aks * blk)

            # ----------------------------------------------------------------
            # Stretch Damping (eq 1.24)
            # C_dot_u = Σ_j ∇Cu_j · v_j
            # f_i = -a * ksd * C_dot_u * ∇Cu_i  (same for v)
            # A velocity Jacobian: +h * a * ksd * outer(gu_i, gu_j)
            # ----------------------------------------------------------------
            v0_qd = state.particle_qd[t0]
            v1_qd = state.particle_qd[t1]
            v2_qd = state.particle_qd[t2]

            Cu_dot = np.dot(gu0, v0_qd) + np.dot(gu1, v1_qd) + np.dot(gu2, v2_qd)
            Cv_dot = np.dot(gv0, v0_qd) + np.dot(gv1, v1_qd) + np.dot(gv2, v2_qd)

            aksd = a * ksd
            if t0_active:
                state.particle_f[t0] -= aksd * (Cu_dot * gu0 + Cv_dot * gv0)
            if t1_active:
                state.particle_f[t1] -= aksd * (Cu_dot * gu1 + Cv_dot * gv1)
            if t2_active:
                state.particle_f[t2] -= aksd * (Cu_dot * gu2 + Cv_dot * gv2)

            haksd = h * aksd
            for idx_i, (ti, gui, gvi, ai) in enumerate(stretch_verts):
                for tj, guj, gvj, aj in stretch_verts[idx_i:]:
                    if ai and aj:
                        i_idx, j_idx = (ti, tj) if ti <= tj else (tj, ti)
                        if ti <= tj:
                            blk = np.outer(gui, guj) + np.outer(gvi, gvj)
                        else:
                            blk = np.outer(guj, gui) + np.outer(gvj, gvi)
                        A.accumu_block(i_idx, j_idx, haksd * blk)

        # ----------------------------------------------------------------
        # Shear Damping (eq 1.24)
        # C_dot = Σ_j ∇C_j · v_j
        # f_i = -a * krd * C_dot * ∇C_i
        # A velocity Jacobian: +h * a * krd * outer(gs_i, gs_j)
        # ----------------------------------------------------------------
        v0_qd = state.particle_qd[t0]
        v1_qd = state.particle_qd[t1]
        v2_qd = state.particle_qd[t2]

        Cs_dot = np.dot(gs0, v0_qd) + np.dot(gs1, v1_qd) + np.dot(gs2, v2_qd)

        akrd = a * krd
        if t0_active:
            state.particle_f[t0] -= akrd * Cs_dot * gs0
        if t1_active:
            state.particle_f[t1] -= akrd * Cs_dot * gs1
        if t2_active:
            state.particle_f[t2] -= akrd * Cs_dot * gs2

        hakrd = h * akrd
        for idx_i, (ti, gi, ai) in enumerate(verts):
            for tj, gj, aj in verts[idx_i:]:
                if ai and aj:
                    i_idx, j_idx = (ti, tj) if ti <= tj else (tj, ti)
                    gi_ord, gj_ord = (gi, gj) if ti <= tj else (gj, gi)
                    A.accumu_block(i_idx, j_idx, hakrd * np.outer(gi_ord, gj_ord))


def eval_cloth_bending_forces(model: Model, state: State) -> None:
    """
    Evaluate the bending forces of the given model, and store the forces
    in `state.particle_f`
    """
    for ii in range(model.edge_count):
        # PDF notation: x1=v0, x2=v1 (edge), x3=v2, x4=v3 (tips)
        v0, v1, v2, v3 = model.edge_indices[ii]
        kb, kbd = model.edge_materials[ii]
        theta_rest = model.edge_rest_angle[ii]

        x1 = state.particle_q[v0]
        x2 = state.particle_q[v1]
        x3 = state.particle_q[v2]
        x4 = state.particle_q[v3]

        e = x2 - x1                        # edge vector
        en = np.linalg.norm(e)

        # area-scaled normals (NOT unit normals)
        n1 = np.cross(e, x3 - x1)          # eq 1.14
        n2 = np.cross(x4 - x1, e)          # eq 1.14

        n1n = np.linalg.norm(n1)
        n2n = np.linalg.norm(n2)

        # skip degenerate triangles
        if en < 1e-10 or n1n < 1e-10 or n2n < 1e-10:
            continue

        # dihedral angle via arctan2 (eq 1.14–1.15)
        cos_t = np.dot(n1, n2) / (n1n * n2n)
        cos_t = np.clip(cos_t, -1.0, 1.0)          # numerical stability
        sin_t = np.dot(np.cross(n1 / n1n, n2 / n2n), e / en)
        theta = np.arctan2(sin_t, cos_t)
        C = theta - theta_rest

        # gradients of theta (eq 1.16–1.19)
        g3 = -(en / n1n ** 2) * n1                              # ∇_{x3} theta
        g4 = -(en / n2n ** 2) * n2                              # ∇_{x4} theta
        g1 = -((x2 - x3) @ e / en ** 2) * g3 - ((x2 - x4) @ e / en ** 2) * g4  # ∇_{x1}
        g2 =  ((x1 - x3) @ e / en ** 2) * g3 + ((x1 - x4) @ e / en ** 2) * g4  # ∇_{x2}

        bend_verts = [(v0, g1), (v1, g2), (v2, g3), (v3, g4)]

        # Bending force: f_i = -kb * C * ∇theta_i  (no area, eq 1.20)
        # NOTE: applied to ALL vertices (including fixed), because bending is explicit.
        # Fixed vertex delta_qd is zeroed out in large_step_cloth.py after solving.
        kbC = kb * C
        for vi, gi in bend_verts:
            state.particle_f[vi] -= kbC * gi

        # Bending damping: f_i = -kbd * C_dot * ∇theta_i
        C_dot = sum(np.dot(gi, state.particle_qd[vi]) for vi, gi in bend_verts)
        kbd_Cdot = kbd * C_dot
        for vi, gi in bend_verts:
            state.particle_f[vi] -= kbd_Cdot * gi
