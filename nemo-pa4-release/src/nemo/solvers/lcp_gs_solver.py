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

        # Integrate without considering contacts.
        self.integrator.integrate(state_in, state_out, f, dt)
        # At this point, state_out has the velocity and position without considering contacts

        # Step 1: compute w_1 and w_2 (mass-inverse in contact space) for each contact
        # w_j = m_j^{-1} + (r_j x n)^T I_world_inv_j (r_j x n)
        # Store per-contact: (w0, w1, r0, I0_inv_world, r1, I1_inv_world)
        # For FIXED_SHAPE_SHAPE: w0=0, r0=None, I0_inv_world=None
        beta = 0.5  # Baumgarte stabilization factor

        contact_data = []
        for i in range(len(contacts.contact_type)):
            ct = contacts.contact_type[i]
            if ct == ContactType.FIXED_SHAPE_SHAPE:
                sid1 = contacts.contact_instance1[i]  # active shape
                n = contacts.contact_normal[i]
                r1 = contacts.contact_point1[i] - state_out.shape_q[sid1, :3]
                R1 = R.from_quat(state_out.shape_q[sid1, 3:], scalar_first=True).as_matrix()
                I1_inv_world = R1 @ self.model.shape_inv_inertia[sid1] @ R1.T
                rxn1 = np.cross(r1, n)
                w1 = self.model.shape_inv_mass[sid1] + rxn1 @ I1_inv_world @ rxn1
                contact_data.append((0.0, w1, None, None, r1, I1_inv_world))

            elif ct == ContactType.SHAPE_SHAPE:
                sid0 = contacts.contact_instance0[i]
                sid1 = contacts.contact_instance1[i]
                n = contacts.contact_normal[i]
                r0 = contacts.contact_point0[i] - state_out.shape_q[sid0, :3]
                r1 = contacts.contact_point1[i] - state_out.shape_q[sid1, :3]
                R0 = R.from_quat(state_out.shape_q[sid0, 3:], scalar_first=True).as_matrix()
                R1 = R.from_quat(state_out.shape_q[sid1, 3:], scalar_first=True).as_matrix()
                I0_inv_world = R0 @ self.model.shape_inv_inertia[sid0] @ R0.T
                I1_inv_world = R1 @ self.model.shape_inv_inertia[sid1] @ R1.T
                rxn0 = np.cross(r0, n)
                rxn1 = np.cross(r1, n)
                w0 = self.model.shape_inv_mass[sid0] + rxn0 @ I0_inv_world @ rxn0
                w1 = self.model.shape_inv_mass[sid1] + rxn1 @ I1_inv_world @ rxn1
                contact_data.append((w0, w1, r0, I0_inv_world, r1, I1_inv_world))

            else:
                contact_data.append(None)

        # Step 2: solve the LCP problem iteratively using Gauss-Seidel
        for _ in range(self.maxits):
            for i in range(len(contacts.contact_type)):
                ct = contacts.contact_type[i]
                if contact_data[i] is None:
                    continue

                n = contacts.contact_normal[i]

                if ct == ContactType.FIXED_SHAPE_SHAPE:
                    w0, w1, _, _, r1, I1_inv_world = contact_data[i]
                    sid1 = contacts.contact_instance1[i]
                    # relative velocity at contact (static body has zero velocity)
                    v1 = state_out.shape_qd[sid1, :3] + np.cross(state_out.shape_qd[sid1, 3:], r1)
                    vi = np.dot(v1, n)

                    if vi >= 0:
                        continue  # receding, no impulse needed

                    # Baumgarte stabilization: bias velocity to push out penetration
                    b = (beta / dt) * contacts.contact_depth[i]
                    fi = (-vi + b) / (w0 + w1)
                    fi = max(0.0, fi)  # impulse must be non-negative

                    impulse = fi * n
                    state_out.shape_qd[sid1, :3] += self.model.shape_inv_mass[sid1] * impulse
                    state_out.shape_qd[sid1, 3:] += I1_inv_world @ np.cross(r1, impulse)

                elif ct == ContactType.SHAPE_SHAPE:
                    w0, w1, r0, I0_inv_world, r1, I1_inv_world = contact_data[i]
                    sid0 = contacts.contact_instance0[i]
                    sid1 = contacts.contact_instance1[i]
                    # relative velocity of body1 w.r.t. body0 at contact
                    v0 = state_out.shape_qd[sid0, :3] + np.cross(state_out.shape_qd[sid0, 3:], r0)
                    v1 = state_out.shape_qd[sid1, :3] + np.cross(state_out.shape_qd[sid1, 3:], r1)
                    vi = np.dot(v1 - v0, n)

                    if vi >= 0:
                        continue  # receding, no impulse needed

                    b = (beta / dt) * contacts.contact_depth[i]
                    fi = (-vi + b) / (w0 + w1)
                    fi = max(0.0, fi)

                    impulse = fi * n
                    if self.model.shape_flags[sid1] & ShapeFlags.ACTIVE.value:
                        state_out.shape_qd[sid1, :3] += self.model.shape_inv_mass[sid1] * impulse
                        state_out.shape_qd[sid1, 3:] += I1_inv_world @ np.cross(r1, impulse)
                    if self.model.shape_flags[sid0] & ShapeFlags.ACTIVE.value:
                        state_out.shape_qd[sid0, :3] -= self.model.shape_inv_mass[sid0] * impulse
                        state_out.shape_qd[sid0, 3:] -= I0_inv_world @ np.cross(r0, impulse)

        # Step 3: update positions and orientations using the corrected velocities
        # Recompute from state_in with the LCP-corrected velocities in state_out
        for i in range(self.model.shape_count):
            if not (self.model.shape_flags[i] & ShapeFlags.ACTIVE.value):
                continue
            # update center of mass position
            state_out.shape_q[i, :3] = state_in.shape_q[i, :3] + state_out.shape_qd[i, :3] * dt
            # update orientation quaternion
            delta_rot = R.from_rotvec(state_out.shape_qd[i, 3:] * dt)
            q_in = R.from_quat(state_in.shape_q[i, 3:], scalar_first=True)
            state_out.shape_q[i, 3:] = (delta_rot * q_in).as_quat(scalar_first=True)
