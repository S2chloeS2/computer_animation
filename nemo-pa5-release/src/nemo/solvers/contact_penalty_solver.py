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


def penalty_force(depth: float, vel: float, stiffness: float, damping: float = 0.0) -> float:
    """
    Args:
        depth: penetration depth, must be POSITIVE
        vel: relative velocity, positive or negative
        stiffness: stiffness coefficient
        damping: damping coefficient

    Returns:
        penalty force
    """
    # vel > 0: receding.
    # vel < 0: approaching.
    return stiffness * depth - damping * vel


class ContactPenaltySolver(SolverBase):
    """Contact solver that uses penalty method to handle contacts."""

    def __init__(
        self,
        model: Model,
        dt: float,
        stiffness: float | None = None,
        damping: float | None = None,
        integrator: IntegratorBase | None = None,
    ):
        """
        Initialize the contact penalty solver.

        Args:
            model: The model to simulate.
            dt: The time step.
            stiffness: The stiffness coefficient for the penalty force.
            damping: The damping coefficient for the penalty force.
            integrator: The time integrator to use.

        NOTE: The stiffness and damping coefficients are used to control the penalty force
              between particles. THe penalty force between particle and fixed shapes is controlled
              by `shape_penalty_params` in the model.
        """
        super().__init__(model=model, dt=dt)

        if integrator is None:
            integrator = SymplecticEulerSolver(model, dt)
        elif integrator.model is not model:
            # check model consistency
            raise RuntimeError("model in the provided solver and the given model must be the same.")
        self.integrator = integrator

        # For contact detection and resolution.
        self.stiffness = stiffness if stiffness is not None else 1000.0
        self.damping = damping if damping is not None else 0.0
        self.collision = CollisionDetector(model)

        rprint("  Use [bold green]Contact Penalty Solver[/bold green]")
        rprint(f"  Particle Contact Penalty: Stiffness={self.stiffness}, Damping={self.damping}")
        if self.model.particle_count > 0:
            mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
            self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)
        if self.model.shape_count > 0:
            mask = self.model.shape_flags & ShapeFlags.ACTIVE.value != 0
            self.masked_shape_mass = np.where(mask, self.model.shape_mass, 0.0)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Step the contact penalty solver.

        Args:
            state_in: The input state.
            state_out: The output state.
            dt: The time step.

        NOTE: this method will detect contacts and evaluate the penalty forces.
        """
        # increase simulated time
        if dt is None:
            dt = self.dt
        self.ts += dt

        # Detect collisions
        contacts = self.collision.instantaneous_contacts(state_in)
        # Evaluate the penalty forces. The contacts are not updated in this function, and thus
        # the penalty force is evaluated only once here.
        f_particle_penalty = np.zeros_like(state_in.particle_f)
        f_shape_penalty = np.zeros_like(state_in.shape_f)

        # f_shape_penalty = np.zeros_like(state_in.shape_f)
        for i in range(len(contacts.contact_type)):
            if contacts.contact_type[i] == ContactType.PARTICLE_PARTICLE:
                p0 = contacts.contact_instance0[i]
                p1 = contacts.contact_instance1[i]
                # relative velocity of p1 w.r.t. p0
                vel = np.dot(state_in.particle_qd[p1] - state_in.particle_qd[p0], contacts.contact_normal[i])
                fp = penalty_force(
                    contacts.contact_depth[i],
                    vel,
                    self.stiffness,
                    self.damping,
                )
                if self.model.particle_flags[p1] & ParticleFlags.ACTIVE.value != 0:
                    f_particle_penalty[p1] += fp * contacts.contact_normal[i]
                if self.model.particle_flags[p0] & ParticleFlags.ACTIVE.value != 0:
                    f_particle_penalty[p0] -= fp * contacts.contact_normal[i]
            elif contacts.contact_type[i] == ContactType.FIXED_SHAPE_PARTICLE:
                shape_id = contacts.contact_instance0[i]
                par_id = contacts.contact_instance1[i]
                vel = np.dot(state_in.particle_qd[par_id], contacts.contact_normal[i])
                fp = penalty_force(
                    contacts.contact_depth[i],
                    vel,
                    self.model.shape_penalty_params[shape_id, 0],
                    self.model.shape_penalty_params[shape_id, 1],
                )
                f_particle_penalty[par_id] += fp * contacts.contact_normal[i]
            elif contacts.contact_type[i] == ContactType.FIXED_SHAPE_SHAPE:
                # the first shape is always fixed, and the second shape is always active
                # velocity at the contact point
                shape_id0 = contacts.contact_instance0[i]
                shape_id1 = contacts.contact_instance1[i]
                r1 = contacts.contact_point1[i] - state_in.shape_q[shape_id1, :3]
                # velocity of shape_id1 at the contact point
                # n.(v + w x r)
                vel = np.dot(
                    state_in.shape_qd[shape_id1, :3] + np.cross(state_in.shape_qd[shape_id1, 3:], r1),
                    contacts.contact_normal[i],
                )
                # use the maximum stiffness and damping parameters between the two shapes
                params = np.maximum(
                    self.model.shape_penalty_params[shape_id0],
                    self.model.shape_penalty_params[shape_id1],
                )
                fp = penalty_force(
                    contacts.contact_depth[i],
                    vel,
                    params[0],
                    params[1],
                )
                # print(f"cp: {contacts.contact_point1[i]}  {contacts.contact_normal[i]} fp={fp}")
                f_shape_penalty[shape_id1, :3] += fp * contacts.contact_normal[i]
                # torque in world frame r x F
                f_shape_penalty[shape_id1, 3:] += fp * np.cross(r1, contacts.contact_normal[i])
            elif contacts.contact_type[i] == ContactType.SHAPE_SHAPE:
                shape_id0 = contacts.contact_instance0[i]
                shape_id1 = contacts.contact_instance1[i]
                r0 = contacts.contact_point1[i] - state_in.shape_q[shape_id0, :3]
                r1 = contacts.contact_point1[i] - state_in.shape_q[shape_id1, :3]
                # rel. velocity of shape_id1 w.r.t. shape_id0 at the contact point
                vel = np.dot(
                    state_in.shape_qd[shape_id1, :3]
                    + np.cross(state_in.shape_qd[shape_id1, 3:], r1)
                    - state_in.shape_qd[shape_id0, :3]
                    - np.cross(state_in.shape_qd[shape_id0, 3:], r0),
                    contacts.contact_normal[i],
                )
                params = np.maximum(
                    self.model.shape_penalty_params[shape_id0],
                    self.model.shape_penalty_params[shape_id1],
                )
                fp = penalty_force(
                    contacts.contact_depth[i],
                    vel,
                    params[0],
                    params[1],
                )
                f_shape_penalty[shape_id1, :3] += fp * contacts.contact_normal[i]
                f_shape_penalty[shape_id1, 3:] += fp * np.cross(r1, contacts.contact_normal[i])
                f_shape_penalty[shape_id0, :3] -= fp * contacts.contact_normal[i]
                f_shape_penalty[shape_id0, 3:] -= fp * np.cross(r0, contacts.contact_normal[i])

        def f(model: Model, state: State) -> None:
            state.clear_forces()
            eval_all_forces(model, state)
            if self.model.particle_count > 0:
                state.particle_f += np.outer(self.masked_mass, self.model.gravity)
                state.particle_f += f_particle_penalty
            if self.model.shape_count > 0:
                state.shape_f[:, :3] += np.outer(self.masked_shape_mass, self.model.gravity)
                state.shape_f += f_shape_penalty

        self.integrator.integrate(state_in, state_out, f, dt)
