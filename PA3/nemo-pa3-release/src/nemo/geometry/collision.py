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

from ..core.types import nparray
from ..geometry.types import GeoType, ParticleFlags
from ..sim.contacts import Contacts, ContactType
from ..sim.model import Model
from ..sim.state import State


def collide_plane_sphere(
    trans: nparray, inv_trans: nparray, pos: nparray, radius: float
) -> tuple[float, nparray, nparray] | None:
    """
    Detect collisions between a plane and a sphere.

    Args:
        trans: The transform matrix from the plane frame to the world frame.
        inv_trans: The inverse transform matrix from the world frame to the plane frame.
        pos: The position of the sphere in the world frame.
        radius: The radius of the sphere.

    Returns:
        - penetration depth: float, negative value indicates penetration
        - contact point on the plane: nparray (3,) in world frame
        - contact point on the sphere: nparray (3,) in world frame

        None if no collision is detected.

    NOTE: The local frame of the plane is defined such that the plane normal is along the y-axis (i.e, the x-z plane).
          `trans` is a 4x4 transform matrix that transforms the plane frame to the world frame.
          `inv_trans` is the inverse of `trans`, which transforms the world frame to the plane frame.
          In this way, the plane normal in world frame is `trans[:3, 1]` (i.e., trans @ np.array([0.0, 1.0, 0.0, 0.0])[:3]).
          and the plane pass throug the point `trans[:3, 3]` (i.e., trans @ np.array([0.0, 0.0, 0.0, 1.0])[:3]).
    """
    # TODO: Implement this function.
    # Replace the following line with your implementation.
    raise NotImplementedError("Not implemented yet.")


def collide_particle_particle(
    pos0: nparray, pos1: nparray, radius0: float, radius1: float
) -> tuple[float, nparray, nparray] | None:
    """
    Detect collisions between two particles.
    Args:
        pos0: The position of the first particle.
        pos1: The position of the second particle.

    Returns:
        - penetration depth: float, negative value indicates penetration
        - contact point on the first particle: nparray (3,) in world frame
        - contact point on the second particle: nparray (3,) in world frame

        None if no collision is detected.
    """
    # TODO: Implement this function.
    # Replace the following line with your implementation.
    raise NotImplementedError("Not implemented yet.")


def continuous_collide_particle_plane(
    trans: nparray,
    inv_trans: nparray,
    pos0: nparray,
    pos1: nparray,
    radius: float,
) -> nparray | None:
    """
    Continuous collision detection between a particle and a plane.

    Args:
        trans: The transform matrix from the plane frame to the world frame.
        inv_trans: The inverse transform matrix from the world frame to the plane frame.
        pos0: The position of the particle at the beginning of the time step.
        pos1: The position of the particle at the end of the time step.
        radius: The radius of the particle.

    Returns:
        - contact normal: nparray (3,) in world frame when the earliest collision happens.
        None if no collision is detected.
    """
    # transform the particle positions to the plane frame
    pos0_plane = inv_trans @ np.append(pos0, 1.0)
    pos1_plane = inv_trans @ np.append(pos1, 1.0)
    depth0 = pos0_plane[1] - radius
    depth1 = pos1_plane[1] - radius
    if (depth1 >= 0.0 and depth0 >= 0.0) or (depth1 >= depth0):
        # Receding contact or no contact, we are fine
        return None
    return trans[:3, 1]


def continuous_collide_particle_particle(
    pos0_in: nparray, pos0_out: nparray, pos1_in: nparray, pos1_out: nparray, radius0: float, radius1: float
) -> tuple[float, nparray] | None:
    """
    Continuous collision detection between two particles.

    Args:
        pos0_in: The position of the first particle at the beginning of the time step.
        pos0_out: The position of the first particle at the end of the time step.
        pos1_in: The position of the second particle at the beginning of the time step.
        pos1_out: The position of the second particle at the end of the time step.
        radius0: The radius of the first particle.
        radius1: The radius of the second particle.

    Returns:
        - earliest collision time: float, must be in the range [0.0, 1.0]
        - contact normal: nparray (3,) contact normal direction in world frame when the earliest collision happens.
        None if no collision is detected.
    """
    # TODO: Implement this function.
    # Replace the following line with your implementation.
    raise NotImplementedError("Not implemented yet.")


class CollisionDetector:
    """Collision detector for the scene."""

    def __init__(self, model: Model):
        self.model = model
        # NOTE: Feel free to add any additional data structures here to facilitate the collision detection.

    def instantaneous_contacts(self, state: State) -> Contacts:
        """
        Detect instantaneous collisions between the objects in the model.
        """
        model = self.model
        contacts = Contacts()
        # Detect contacts between particles and fixed shapes.
        # NOTE: we didn't optimize the collision detection for particles and shapes,
        #       since the number of particles and shapes are usually small.
        for j in range(model.particle_count):
            if model.particle_flags[j] & ParticleFlags.ACTIVE.value == 0:
                # ignore fixed particles
                continue
            for i in range(model.shape_count):
                if model.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                    ret = collide_plane_sphere(
                        model.shape_transform[i],
                        model.shape_inv_transform[i],
                        pos=state.particle_q[j],
                        radius=model.particle_radius[j],
                    )
                    if ret is not None:
                        depth, contact_point_plane, contact_point_sphere = ret
                        if depth < 0.0:
                            contacts.contact_instance0.append(i)
                            contacts.contact_instance1.append(j)
                            contacts.contact_type.append(ContactType.FIXED_SHAPE_PARTICLE)
                            contacts.contact_point0.append(contact_point_plane)
                            contacts.contact_point1.append(contact_point_sphere)
                            # NOTE: the contact normal is the normal in world frame.
                            # pointing out of the FIRST object, which is the fixed plane here.
                            contacts.contact_normal.append(model.shape_transform[i, :3, 1])

        # Detect contacts between particles.
        # TODO: Implement collision detection for particle-particle contact.
        # The naive algorithm compares all pairs of particles, which is O(N^2) in the number of particles.
        # You should try to implment the algorithm at least in O(N log N) complexity that we discussed in the
        # lecture.

        return contacts

    def continuous_contacts(self, state_0: State, state_1: State, contacts: Contacts) -> None:
        """
        Detect continuous collisions between the objects in the model.

        Args:
            state_0: The state at the beginning of the time step.
            state_1: The state at the end of the time step.
            contacts: The contacts object to store the continuous contacts.

        NOTE: Different from `instantaneous_contacts`, this method will update
        the continuous contacts in the contacts object. This is because `continuous_contacts`
        is designed to be called in contact iterations. In each iteration, we need to update
        the continuous contacts based on the previous state and the current state. Instead of
        creating a new contacts object every time, we update the existing contacts object inplace.
        """
        contacts.clear_continuous_contacts()
        model = self.model

        for j in range(model.particle_count):
            if model.particle_flags[j] & ParticleFlags.ACTIVE.value == 0:
                # ignore fixed particles
                continue
            for i in range(model.shape_count):
                if model.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                    # TODO: Use `continuous_collide_particle_plane` to detect collision between the particle and the plane.
                    # And add the detected contact information to the `contacts` object.

                    # Replace the following line with your implementation.
                    pass

        # TODO: Implement continuous collision detection for particle-particle contact.
        # Detect contacts between particles.
        # HERE it's ok if you implement a O(N^2) algorithm.
        # The implmentation here is just to use the `continuous_collide_particle_particle` function
        # and add detected constacts in `contacts` object.
