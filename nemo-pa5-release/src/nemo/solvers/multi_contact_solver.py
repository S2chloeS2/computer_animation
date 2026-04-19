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
from .contact_penalty_solver import penalty_force
from .integrator import IntegratorBase
from .solver import SolverBase
from .symplectic_euler import SymplecticEulerSolver


class MultiContactSolver(SolverBase):
    """Contact solver that considers multiple contacts.

    This solver responds to contacts using an iterative approach that we discussed in class.
    """

    def __init__(
        self,
        model: Model,
        dt: float,
        maxits: int | None = None,
        stiffness: float | None = None,
        damping: float | None = None,
        integrator: IntegratorBase | None = None,
    ):
        """
        Initialize the iterative contact solver.

        Args:
            model: The model to simulate.
            dt: The time step.
            maxits: The maximum number of iterations to resolve contact before resorting to the
                    gemetric collision response.
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

        if maxits is None:
            maxits = 8
        elif maxits < 0:
            raise ValueError("maxits must be non-negative")

        self.maxits = maxits
        self.stiffness = stiffness if stiffness is not None else 1000.0
        self.damping = damping if damping is not None else 0.0
        rprint("  Use [bold green]Multi-Contact Solver[/bold green]")
        rprint(f"  Particle Contact Penalty: Stiffness={self.stiffness}, Damping={self.damping}")
        rprint(f"  Multi-Contact Solver: maxits={self.maxits}")
        self.collision = CollisionDetector(model)

        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """Step the contact with an iterative solver.

        Args:
            state_in: The input state.
            state_out: The output state.
            contacts: The contact information.
            dt: The time step.

        NOTE: the contacts must be updated before calling this method
              by the collision detection algorithm (see geometry/collision.py).
              The contacts must be updated based on state_in (not state_out).
        """
        if dt is None:
            dt = self.dt
        self.ts += dt

        # Detect collisions
        contacts = self.collision.instantaneous_contacts(state_in)
        # First, we integrate the system by considering instantaneous contacts and penalty forces.
        # This is exactly the same as the ContactPenaltySolver.

        # Evaluate the penalty forces. The contacts are not updated in this function, and thus
        # the penalty force is evaluated only once here.
        f_penalty = np.zeros_like(state_in.particle_f)
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
                    f_penalty[p1] += fp * contacts.contact_normal[i]
                if self.model.particle_flags[p0] & ParticleFlags.ACTIVE.value != 0:
                    f_penalty[p0] -= fp * contacts.contact_normal[i]
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
                f_penalty[par_id] += fp * contacts.contact_normal[i]

        def f_pen(model: Model, state: State) -> None:
            state.clear_forces()
            eval_all_forces(model, state)
            state.particle_f += np.outer(self.masked_mass, self.model.gravity)
            # include penalty forces
            state.particle_f += f_penalty

        self.integrator.integrate(state_in, state_out, f_pen, dt)

        # Next, we enter into iterative solver
        for _ in range(self.maxits):
            # 1. Continuous collision detection
            self.collision.continuous_contacts(state_in, state_out, contacts)
            # If there are no continuous contacts, we can exit the loop
            if len(contacts.contact_type_continuous) == 0:
                # print(f"BREAK t={t}")
                break
            # 2. Apply impulses
            for i in range(len(contacts.contact_type_continuous)):
                if contacts.contact_type_continuous[i] == ContactType.PARTICLE_PARTICLE:
                    p0 = contacts.contact_instance0_continuous[i]
                    p1 = contacts.contact_instance1_continuous[i]
                    v0 = state_out.particle_q[p0] - state_in.particle_q[p0]
                    v1 = state_out.particle_q[p1] - state_in.particle_q[p1]
                    denom = self.model.particle_inv_mass[p0] + self.model.particle_inv_mass[p1]
                    vel_rel_nrm = np.dot(v1 - v0, contacts.contact_normal_continuous[i])
                    if denom > 0.0 and vel_rel_nrm < 0.0:
                        restitution_coeff = min(
                            self.model.particle_restitution_coeff[p0], self.model.particle_restitution_coeff[p1]
                        )
                        imp = contacts.contact_normal_continuous[i] * (vel_rel_nrm * (1.0 + restitution_coeff) / denom)
                        if self.model.particle_flags[p0] & ParticleFlags.ACTIVE.value != 0:
                            state_out.particle_q[p0] += imp
                            state_out.particle_qd[p0] += imp / dt
                        if self.model.particle_flags[p1] & ParticleFlags.ACTIVE.value != 0:
                            state_out.particle_q[p1] -= imp
                            state_out.particle_qd[p1] -= imp / dt

                elif contacts.contact_type_continuous[i] == ContactType.FIXED_SHAPE_PARTICLE:
                    shape_id = contacts.contact_instance0_continuous[i]
                    par_id = contacts.contact_instance1_continuous[i]
                    restitution_coeff = min(
                        self.model.shape_restitution_coeff[shape_id], self.model.particle_restitution_coeff[par_id]
                    )
                    vel_est = state_out.particle_q[par_id] - state_in.particle_q[par_id]
                    vel_est_nrm = np.dot(vel_est, contacts.contact_normal_continuous[i])
                    # assert vel_est_nrm <= 0.0, f"vnorm={vel_est_nrm}, [{len(contacts.contact_type_continuous)}]"
                    if vel_est_nrm < 0.0:
                        imp = contacts.contact_normal_continuous[i] * (-vel_est_nrm * (1.0 + restitution_coeff))
                        state_out.particle_q[par_id] += imp
                        state_out.particle_qd[par_id] += imp / dt
