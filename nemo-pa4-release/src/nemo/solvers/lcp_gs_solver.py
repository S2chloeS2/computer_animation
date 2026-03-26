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

        # TODO: Finish the implementation of the LCP Gauss-Seidel solver.
        # You need to implement the following steps:
        # 1. Compute w_1 and w_2 for each contact point.
        # 2. Solve the LCP problem iteratively
        # 3. Update the velocity and position of the rigid bodies.
        #
        # Please refer to Sec. 1.2.2 of the course notes.

        # Step 1: comptue w_1 and w_2
        # for i in range(len(contacts.contact_type)):
        # ...

        # Step 2: solve the LCP problem iteratively
        # for _ in range(self.maxits):
        #     for i in range(len(contacts.contact_type)):
        #         ...

        # Step 3: integrate with the new velocity
        # This is very much the same what you implemented in the symplectic time integrator.
        # (i.e., the part 1 of this assignment)
