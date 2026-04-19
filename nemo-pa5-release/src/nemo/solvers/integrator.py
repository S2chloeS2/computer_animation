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

from collections.abc import Callable

from ..sim.model import Model
from ..sim.state import State


class IntegratorBase:
    """Base class for integrators."""

    def __init__(self, model: Model):
        self.model = model

    def integrate(self, state_in: State, state_out: State, f: Callable[[Model, State], None], dt: float) -> None:
        """
        Step the integrator.

        Args:
            state_in: The input state.
            state_out: The output state.
            f: The function to evaluate the force, and store the forces in the state.particle_f.
            dt: The time step.

        NOTE: The implementation of this method may or may not clear the forces in the
              state.particle_f before calling the function f.
        """
        raise NotImplementedError("IntegratorBase.step is not implemented")
