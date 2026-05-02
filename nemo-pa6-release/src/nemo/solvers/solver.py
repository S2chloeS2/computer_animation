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
from ..sim.model import Model
from ..sim.state import State


class SolverBase:
    """Generic base class for solvers."""

    def __init__(self, model: Model, dt: float):
        self.model = model
        self.dt = dt
        """Default timestep size."""
        self.ts = 0.0
        """Accumulated time that has been stepped"""

    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """
        Simulate the model for a given time step.

        Args:
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).

        NOTE:
            When dt is None, this step call will use the default timestep size
            stored in self.dt. Otherwise, the given dt will be used.
        """
        raise NotImplementedError()
