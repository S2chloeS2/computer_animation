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

from enum import IntEnum


class ContactType(IntEnum):
    """Type of contact."""

    PARTICLE_PARTICLE = 0
    """Contact between two particles."""
    FIXED_SHAPE_PARTICLE = 1
    """Contact between a fixed shape and a particle."""


class Contacts:
    """Store contact information for rigid/soft body and particles."""

    def __init__(self):
        """
        Instantaneous contact information:
            - contact_instance0: list[int] (N,): index of the first object in the contact
            - contact_instance1: list[int] (N,): index of the second object in the contact
            - contact_point0: nparray (N, 3): in world frame
            - contact_point1: nparray (N, 3): in world frame
            - contact_normal: nparray (N, 3): normal direction at contact point expressed in world frame

        NOTE: the noraml is always pointing toward the outside of the first object.
              the penatration depth is then: (contact_point0 - contact_point1) dot contact_normal, a positive
              value indicates penetration.
        """
        # -------------------------------------------------------------------------
        # Data structures for storing instantaneous contacts.
        self.contact_instance0: list[int] = []
        self.contact_instance1: list[int] = []
        self.contact_type: list[ContactType] = []  # type: list[ContactType]
        self.contact_point0 = []
        self.contact_point1 = []
        self.contact_normal = []

        # -------------------------------------------------------------------------
        # Data structures for storing continuous contacts.
        self.contact_instance0_continuous: list[int] = []
        self.contact_instance1_continuous: list[int] = []
        self.contact_type_continuous: list[ContactType] = []
        self.contact_normal_continuous = []

    def clear_continuous_contacts(self) -> None:
        """
        Clear the continuous contact information.

        This method will be called before the continuous collision detection.
        (see `CollisionDetector.continuous_contacts`)
        """
        self.contact_instance0_continuous.clear()
        self.contact_instance1_continuous.clear()
        self.contact_type_continuous.clear()
        self.contact_normal_continuous.clear()
