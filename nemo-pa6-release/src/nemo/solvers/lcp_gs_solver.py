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
from rich import print as rprint
from scipy.spatial.transform import Rotation as R

from ..core.types import override
from ..geometry.collision import CollisionDetector
from ..geometry.types import ParticleFlags, ShapeFlags
from ..sim.contacts import ContactType
from ..sim.forces import eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .integrator import IntegratorBase
from .solver import SolverBase
from .symplectic_euler import SymplecticEulerSolver


class LCPGSSolver(SolverBase):
    """A simple Gauss-Seidel solver for linear complementarity problems
    arised from rigid body contact.
    """

    def __init__(
        self,
        model: Model,
        dt: float,
        integrator: IntegratorBase | None = None,
        maxits: int = 5,
    ):
        """
        Initialize the LCP Gauss-Seidel solver.

        Args:
            model: The model to simulate.
            dt: The time step.
            integrator: The time integrator to use.
            maxits: The maximum number of iterations for the Gauss-Seidel solver.
        """
        super().__init__(model=model, dt=dt)

        if integrator is None:
            integrator = SymplecticEulerSolver(model, dt)
        elif integrator.model is not model:
            # check model consistency
            raise RuntimeError("model in the provided solver and the given model must be the same.")

        self.integrator = integrator
        self.collision = CollisionDetector(model)
        rprint("  Use [bold green]LCP Gauss-Seidel Solver[/bold green]")

        if self.model.particle_count > 0:
            mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
            self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)
        if self.model.shape_count > 0:
            mask = self.model.shape_flags & ShapeFlags.ACTIVE.value != 0
            self.masked_shape_mass = np.where(mask, self.model.shape_mass, 0.0)
        # maximum number of iterations for the Gauss-Seidel solver
        self.maxits = maxits

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Step the solver

        Args:
            state_in: The input state.
            state_out: The output state.
            dt: The time step.
        """
        # increase simulated time
        if dt is None:
            dt = self.dt
        self.ts += dt

        # Detect collisions
        contacts = self.collision.instantaneous_contacts(state_in)

        def f(model: Model, state: State) -> None:
            state.clear_forces()
            eval_all_forces(model, state)
            if self.model.particle_count > 0:
                state.particle_f += np.outer(self.masked_mass, self.model.gravity)
            if self.model.shape_count > 0:
                state.shape_f[:, :3] += np.outer(self.masked_shape_mass, self.model.gravity)

        # integrate without considering contacts
        # state_out has the velocity and position without considering contacts
        self.integrator.integrate(state_in, state_out, f, dt)

        model = self.model
        # solve the linear complementarity problem for the contacts
        w = np.empty(shape=(len(contacts.contact_type), 2), dtype=np.float64)  # to store w_i
        # Is cache the inverse of the inertia tensor in the world frame
        Is = [None] * model.shape_count
        for i in range(len(contacts.contact_type)):
            if contacts.contact_type[i] == ContactType.FIXED_SHAPE_SHAPE:
                id1 = contacts.contact_instance1[i]
                r1 = contacts.contact_point1[i] - state_in.shape_q[id1, :3]
                if Is[id1] is None:
                    rot = R.from_quat(state_in.shape_q[id1, 3:], scalar_first=True).as_matrix()
                    Is[id1] = rot @ model.shape_inv_inertia[id1] @ rot.T
                rn = np.cross(r1, contacts.contact_normal[i])
                w[i, 1] = model.shape_inv_mass[id1] + np.dot(rn, Is[id1] @ rn)
            elif contacts.contact_type[i] == ContactType.SHAPE_SHAPE:
                id0 = contacts.contact_instance0[i]
                id1 = contacts.contact_instance1[i]
                r0 = contacts.contact_point1[i] - state_in.shape_q[id0, :3]
                r1 = contacts.contact_point1[i] - state_in.shape_q[id1, :3]
                if Is[id0] is None:
                    rot = R.from_quat(state_in.shape_q[id0, 3:], scalar_first=True).as_matrix()
                    Is[id0] = rot @ model.shape_inv_inertia[id0] @ rot.T
                if Is[id1] is None:
                    rot = R.from_quat(state_in.shape_q[id1, 3:], scalar_first=True).as_matrix()
                    Is[id1] = rot @ model.shape_inv_inertia[id1] @ rot.T
                rn0 = np.cross(r0, contacts.contact_normal[i])
                rn1 = np.cross(r1, contacts.contact_normal[i])
                w[i, 0] = model.shape_inv_mass[id0] + np.dot(rn0, Is[id0] @ rn0)
                w[i, 1] = model.shape_inv_mass[id1] + np.dot(rn1, Is[id1] @ rn1)

        # LCP Gauss-Seidel solver
        for _ in range(self.maxits):
            cnt = 0
            for i in range(len(contacts.contact_type)):
                if contacts.contact_type[i] == ContactType.FIXED_SHAPE_SHAPE:
                    id1 = contacts.contact_instance1[i]
                    r1 = contacts.contact_point1[i] - state_in.shape_q[id1, :3]
                    # n.(v + w x r)
                    vel = np.dot(
                        state_out.shape_qd[id1, :3] + np.cross(state_out.shape_qd[id1, 3:], r1),
                        contacts.contact_normal[i],
                    )
                    if vel < 0.0:
                        cnt += 1
                        # Since you solve for velocity at the current frame to
                        # be zero, gravity immediately accelerates the object
                        # during the next integration step, pushing it into the
                        # ground before the solver gets another chance to look
                        # at it.
                        #
                        # The most common fix (used in Box2D, # Bullet, etc.). Instead of
                        # solving for a target relative velocity of $0$, you
                        # solve for a small "rebound" velocity that is
                        # proportional to how deep the penetration already is.
                        bias = contacts.contact_depth[i] * 0.5 / dt  # bias velocity to prevent penetration
                        f = (-vel + bias) / w[i, 1]  # impulse
                        state_out.shape_qd[id1, :3] += model.shape_inv_mass[id1] * f * contacts.contact_normal[i]
                        state_out.shape_qd[id1, 3:] += Is[id1] @ (f * np.cross(r1, contacts.contact_normal[i]))
                elif contacts.contact_type[i] == ContactType.SHAPE_SHAPE:
                    id0 = contacts.contact_instance0[i]
                    id1 = contacts.contact_instance1[i]
                    r0 = contacts.contact_point1[i] - state_in.shape_q[id0, :3]
                    r1 = contacts.contact_point1[i] - state_in.shape_q[id1, :3]
                    # relative velocity of shape1 w.r.t. shape0 at the contact point
                    vel = np.dot(
                        state_out.shape_qd[id1, :3]
                        + np.cross(state_out.shape_qd[id1, 3:], r1)
                        - state_out.shape_qd[id0, :3]
                        - np.cross(state_out.shape_qd[id0, 3:], r0),
                        contacts.contact_normal[i],
                    )
                    if vel < 0.0:
                        cnt += 1
                        bias = contacts.contact_depth[i] * 0.5 / dt  # bias velocity to prevent penetration
                        f = (-vel + bias) / (w[i, 1] + w[i, 0])  # impulse
                        state_out.shape_qd[id1, :3] += model.shape_inv_mass[id1] * f * contacts.contact_normal[i]
                        state_out.shape_qd[id1, 3:] += Is[id1] @ (f * np.cross(r1, contacts.contact_normal[i]))
                        state_out.shape_qd[id0, :3] -= model.shape_inv_mass[id0] * f * contacts.contact_normal[i]
                        state_out.shape_qd[id0, 3:] -= Is[id0] @ (f * np.cross(r0, contacts.contact_normal[i]))

            if cnt == 0:
                break

        # integrate with the new velocity
        state_out.shape_q[:, :3] = state_in.shape_q[:, :3] + state_out.shape_qd[:, :3] * dt
        r0 = R.from_quat(state_in.shape_q[:, 3:], scalar_first=True)  # current pose
        state_out.shape_q[:, 3:] = (R.from_rotvec(state_out.shape_qd[:, 3:] * dt) * r0).as_quat(scalar_first=True)
