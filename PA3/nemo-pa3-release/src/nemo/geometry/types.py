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
from enum import IntEnum


# Particle flags
class ParticleFlags(IntEnum):
    """
    Flags for particle properties.
    """

    ACTIVE = 1 << 0
    """Indicates that the particle is active."""


class GeoType(IntEnum):
    HALF_SPACE_PLANE = 0
    """Half space plane."""
    BOX = 1
    """Axis-aligned box."""
