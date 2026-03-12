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
class PlotSpec:
    def __init__(self, config: dict):
        self.particle_id = config["particle_id"]
        self.dof = config["dof"]
        v = config["y_range"]
        self.y_range_min = v[0]
        self.y_range_max = v[1]
