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
from scipy.spatial.transform import Rotation as R

from ..core.types import nparray


def body_q_to_transform(body_q: nparray) -> nparray:
    """
    Convert a body quaternion to a transform matrix.
    """
    trans = np.eye(4)
    trans[:3, :3] = R.from_quat(body_q[3:], scalar_first=True).as_matrix()
    trans[:3, 3] = body_q[:3]
    return trans
