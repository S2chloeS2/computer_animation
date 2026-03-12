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
    # Plane normal (unit) and origin in world frame
    plane_normal = trans[:3, 1]
    plane_origin = trans[:3, 3]

    # Signed distance from sphere center to plane surface (along normal)
    dist_to_plane = np.dot(pos - plane_origin, plane_normal)
    depth = dist_to_plane - radius

    # Contact point on the plane: projection of sphere center onto the plane
    contact_point_plane = pos - dist_to_plane * plane_normal

    # Contact point on the sphere: surface point closest to the plane
    contact_point_sphere = pos - radius * plane_normal

    return depth, contact_point_plane, contact_point_sphere


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
    diff = pos1 - pos0
    dist = np.linalg.norm(diff)

    # Degenerate case: same position, normal is undefined
    if dist < 1e-12:
        return None

    n_hat = diff / dist
    depth = dist - (radius0 + radius1)

    contact_point0 = pos0 + radius0 * n_hat
    contact_point1 = pos1 - radius1 * n_hat

    return depth, contact_point0, contact_point1


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
    # Relative displacement: delta = (pos1_out - pos1_in) - (pos0_out - pos0_in)
    d = pos1_in - pos0_in           # initial separation vector
    delta = (pos1_out - pos1_in) - (pos0_out - pos0_in)

    dd = np.dot(delta, delta)       # |delta|^2 (squared relative speed)

    if dd < 1e-12:
        # No relative motion between the two particles
        return None

    d_dot_delta = np.dot(d, delta)
    d_dot_d = np.dot(d, d)
    r_sum_sq = (radius0 + radius1) ** 2

    # Discriminant of quadratic p(t) = (r0+r1)^2 - n(t)·n(t) > 0
    # p(t) > 0 between roots t0 and t1 (particles overlapping)
    disc = d_dot_delta * d_dot_delta + dd * (r_sum_sq - d_dot_d)

    if disc < 0:
        return None  # Never overlapping within this timestep

    sqrt_disc = np.sqrt(disc)
    t0 = (-d_dot_delta - sqrt_disc) / dd  # Earlier root: collision begins
    t1 = (-d_dot_delta + sqrt_disc) / dd  # Later root:   collision ends

    # At t0: relative normal velocity g(t0) = sqrt_disc / dd >= 0 (approaching)
    # → no separate approach check needed when tc = t0

    if 0.0 <= t0 <= 1.0:
        # Normal case: collision begins in [0, 1]
        tc = t0
    elif t0 < 0.0 <= t1:
        # Already overlapping at t=0; only valid if particles are still approaching
        # g(0) = -(d · delta) > 0  ↔  d_dot_delta < 0
        if d_dot_delta < 0.0:
            tc = 0.0
        else:
            return None  # Overlapping but receding
    else:
        # t0 > 1 (collision starts after window) or t1 < 0 (entirely in the past)
        return None

    # Contact normal at tc: from particle 0 toward particle 1
    n_tc = d + tc * delta
    n_len = np.linalg.norm(n_tc)
    if n_len < 1e-12:
        return None

    return tc, n_tc / n_len


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

        # Detect contacts between particles using sort-and-sweep (O(N log N)).
        # Sort particle indices by their minimum x-extent (pos_x - radius).
        # Then sweep: for each particle i, only test particles j whose min x-extent
        # has not yet passed i's max x-extent, giving an early-exit inner loop.
        positions = state.particle_q
        radii = model.particle_radius
        flags = model.particle_flags

        sorted_indices = np.argsort(positions[:, 0] - radii)

        for ii in range(model.particle_count):
            i = sorted_indices[ii]
            max_xi = positions[i, 0] + radii[i]
            for jj in range(ii + 1, model.particle_count):
                j = sorted_indices[jj]
                # Early exit: no further particle can overlap i along x
                if positions[j, 0] - radii[j] > max_xi:
                    break
                # Skip fixed-fixed pairs (neither will respond)
                if (flags[i] & ParticleFlags.ACTIVE.value == 0) and (flags[j] & ParticleFlags.ACTIVE.value == 0):
                    continue
                ret = collide_particle_particle(positions[i], positions[j], radii[i], radii[j])
                if ret is not None:
                    depth, cp0, cp1 = ret
                    if depth < 0.0:
                        # Normal points outward from particle i (object0) toward particle j
                        diff = positions[j] - positions[i]
                        n_hat = diff / np.linalg.norm(diff)
                        contacts.contact_instance0.append(i)
                        contacts.contact_instance1.append(j)
                        contacts.contact_type.append(ContactType.PARTICLE_PARTICLE)
                        contacts.contact_point0.append(cp0)
                        contacts.contact_point1.append(cp1)
                        contacts.contact_normal.append(n_hat)

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
