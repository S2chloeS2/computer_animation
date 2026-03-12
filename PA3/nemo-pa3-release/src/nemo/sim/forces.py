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
