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
from ..geometry.types import ParticleFlags
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
        depth: penetration depth, must be negative
        vel: relative velocity, positive or negative
        stiffness: stiffness coefficient
        damping: damping coefficient

    Returns:
        penalty force
    """
    # vel > 0: receding.
    # vel < 0: approaching.
    return -stiffness * depth - damping * vel


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
        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)

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
        # - f_penalty is the penalty force applied to the particles, which will be computed in the 
        #   following loop.
        f_penalty = np.zeros_like(state_in.particle_f)
        for i in range(len(contacts.contact_type)):
            if contacts.contact_type[i] == ContactType.PARTICLE_PARTICLE:
                par0_id = contacts.contact_instance0[i]
                par1_id = contacts.contact_instance1[i]
                n = contacts.contact_normal[i]
                depth = np.dot(contacts.contact_point1[i] - contacts.contact_point0[i], n)
                if depth < 0.0:
                    # Relative velocity along normal: positive = receding, negative = approaching
                    v_rel = np.dot(state_in.particle_qd[par1_id] - state_in.particle_qd[par0_id], n)
                    F = penalty_force(depth, v_rel, self.stiffness, self.damping)
                    # Equal and opposite forces; skip fixed particles
                    if self.model.particle_flags[par1_id] & ParticleFlags.ACTIVE.value != 0:
                        f_penalty[par1_id] += F * n
                    if self.model.particle_flags[par0_id] & ParticleFlags.ACTIVE.value != 0:
                        f_penalty[par0_id] -= F * n
            elif contacts.contact_type[i] == ContactType.FIXED_SHAPE_PARTICLE:
                shape_id = contacts.contact_instance0[i]
                par_id = contacts.contact_instance1[i]
                depth = np.dot(contacts.contact_point1[i] - contacts.contact_point0[i], contacts.contact_normal[i])
                if depth < 0.0:
                    n = contacts.contact_normal[i]
                    stiffness = self.model.shape_penalty_params[shape_id, 0]
                    damping = self.model.shape_penalty_params[shape_id, 1]
                    # Velocity of particle along the outward plane normal
                    v_rel = np.dot(state_in.particle_qd[par_id], n)
                    F = penalty_force(depth, v_rel, stiffness, damping)
                    f_penalty[par_id] += F * n

        def f(model: Model, state: State) -> None:
            eval_all_forces(model, state)
            state.particle_f += np.outer(self.masked_mass, self.model.gravity)
            # include penalty forces
            state.particle_f += f_penalty

        self.integrator.integrate(state_in, state_out, f, dt)
