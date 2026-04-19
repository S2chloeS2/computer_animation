#
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Written by Changxi Zheng <cxz@cs.columbia.edu>, 2026
#
import numpy as np

from nemo.sim import ModelBuilder
from nemo.sim.forces import eval_gravitational_forces


def test_gravitational_forces():
    builder = ModelBuilder()
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_particle(pos=(1, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_gravitational(0, 1, 1.0)
    model = builder.finalize()
    state = model.state()
    eval_gravitational_forces(model, state)
    assert np.all(state.particle_f[0] == np.array([1.0, 0.0, 0.0]))
    assert np.all(state.particle_f[1] == np.array([-1.0, 0.0, 0.0]))
