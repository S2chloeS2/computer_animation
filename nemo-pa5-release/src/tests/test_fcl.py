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

import fcl
import numpy as np
import pytest


def test_fcl_simple():
    s1 = fcl.Sphere(4.0)
    s2 = fcl.Sphere(4.0)
    c1 = fcl.CollisionObject(s1)
    c2 = fcl.CollisionObject(s2)
    c1.setTranslation(np.array([0.0, 7.8, 0.0]))

    request = fcl.CollisionRequest()
    request.enable_contact = True
    # print(f"enable_cost: {request.enable_cost}") #request.enable_cost = False
    # print(f"gjk_type: {request.gjk_solver_type}") # = fcl.GJKSolverType.GST_LIBCCD
    result = fcl.CollisionResult()
    ret = fcl.collide(c1, c2, request, result)
    assert ret == 1
    assert result.contacts[0].penetration_depth == pytest.approx(0.2)

    h = fcl.Halfspace(np.array([0.0, 1.0, 0.0]), -3.9)
    c3 = fcl.CollisionObject(h)
    # c3.setTranslation(np.array([0.0, 0.0, 0.0]))
    ret = fcl.collide(c2, c3, request, result)
    assert ret == 1
    # print(f"depth: {result.contacts[0].penetration_depth}")
    assert result.contacts[0].penetration_depth == pytest.approx(0.2)


def test_fcl_box():
    s1 = fcl.Box(2.0, 1.0, 3.0)
    c1 = fcl.CollisionObject(s1)
    c1.setTranslation(np.array([2.0, 0.0, 3.0]))

    h = fcl.Halfspace(np.array([0.0, 1.0, 0.0]), -0.48)
    c2 = fcl.CollisionObject(h)

    request = fcl.CollisionRequest()
    request.enable_contact = True

    result = fcl.CollisionResult()
    ret = fcl.collide(c2, c1, request, result)
    assert ret == 1
    # for c in result.contacts:
    #    print(c.normal)
    #    print(c.penetration_depth)
    #    print(c.pos)
