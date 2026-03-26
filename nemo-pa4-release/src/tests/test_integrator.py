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

from nemo.sim import ModelBuilder
from nemo.solvers import SymplecticEulerSolver


def test_integrator_0():
    builder = ModelBuilder()
    builder.add_shape_cylinder([0.0, 0.0, 1.0], 1.0, 1.0, [1.0, 0.0, 0.0, 0.0])
    builder.add_shape_half_space_plane([0.0, 0.0, 1.0], -2.0)
    model = builder.finalize()

    state_0 = model.state()
    state_1 = model.state()
    integrator = SymplecticEulerSolver(model, 0.01)
    integrator.step(state_0, state_1)
    # print(state_0.shape_q)
    assert np.allclose(state_1.shape_q[0, 3:], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(state_1.shape_q[1], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
