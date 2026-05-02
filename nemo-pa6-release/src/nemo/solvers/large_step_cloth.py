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
import scipy.sparse.linalg as sparse_linalg

from ..core.types import override
from ..geometry.types import ParticleFlags
from ..sim.forces import eval_cloth_bending_forces, eval_cloth_stretch_shear_forces
from ..sim.model import Model
from ..sim.sparse import JacobiParticlePrecond, ParticleIndexCache, SparseParticleSystem
from ..sim.state import State
from .solver import SolverBase


class LargeStepClothSolver(SolverBase):
    """This is a solver for cloth simulation.
    This solver is based on the paper "Large Step Cloth Simulation" by David Baraff and Andrew Witkin.
    (see https://www.cs.cmu.edu/~baraff/papers/sig98.pdf)
    """

    def __init__(self, model: Model, dt: float):
        """Initialize the large step cloth solver."""
        super().__init__(model=model, dt=dt)
        if self.model.particle_count > 0:
            self.mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
            self.masked_mass = np.where(self.mask, self.model.particle_mass, 0.0)
            self.particle_index_cache = ParticleIndexCache(model)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate cloth particles for a given time step.
        """
        if dt is None:
            dt = self.dt
        self.ts += dt

        model = self.model
        if model.particle_count == 0:
            return

        # The high-level algorithm here is similar to linearized implicit euler.
        # However, before we time step, we need to detect collisions and apply collision impulses
        # to change particle velocities.
        np.copyto(state_out.particle_q, state_in.particle_q)
        np.copyto(state_out.particle_qd, state_in.particle_qd)
        state_out.clear_forces()
        # evaluate Bending forces
        # Bending forces will be used in an explicit fashion, and thus evaluated at the beginning of the timestep.
        eval_cloth_bending_forces(model, state_out)
        state_out.particle_f += np.outer(self.masked_mass, self.model.gravity)
        # apply contact impulses (Not needed for this assignment)

        # Then, we need to perform the following steps:
        # 1. advance position only
        state_out.particle_q = state_out.particle_q + state_out.particle_qd * dt
        # state_out now holds (q^n + h * qd^n, qd^n)

        A = SparseParticleSystem(model, self.particle_index_cache)
        # 2. evalue force at new position
        # Evaluate Stretch and Shear forces and populate A matrix
        eval_cloth_stretch_shear_forces(model, state_out, A, dt)
        # 3. solve the linear system
        # Construct a simple diagonal preconditioner
        precond = JacobiParticlePrecond(A)
        # 3.2 solve the linear system using PCG
        delta_qd, exit_code = sparse_linalg.cg(A, state_out.particle_f.reshape(-1) * dt, M=precond, maxiter=30)
        if exit_code != 0:
            print(f"PCG failed (its = {exit_code})")

        # 4. update the velocity and position
        delta_qd = delta_qd.reshape(-1, 3)
        delta_qd[~self.mask] = 0.0
        state_out.particle_qd += delta_qd
        state_out.particle_q = state_in.particle_q + state_out.particle_qd * dt

        # Finally, fix the positions of collision particles.
        # (Not needed for this assignment)
