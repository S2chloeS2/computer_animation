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

from ..core.types import nparray
from ..geometry.types import GeoType, ParticleFlags, ShapeFlags
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
          In this way, the plane normal in world frame is `trans[:3, 1]`
          (i.e., trans @ np.array([0.0, 1.0, 0.0, 0.0])[:3]).
          and the plane pass throug the point `trans[:3, 3]` (i.e., trans @ np.array([0.0, 0.0, 0.0, 1.0])[:3]).
    """
    # transform the sphere position to the plane frame
    pos_plane = inv_trans @ np.append(pos, 1.0)
    # check if the sphere is in the plane
    depth = pos_plane[1] - radius
    if depth >= 0.0:
        return None
    # calculate the contact point on the plane
    pos_plane[1] = 0.0
    contact_point_plane = trans @ pos_plane
    # calculate the contact point on the sphere
    pos_plane[1] = depth
    contact_point_sphere = trans @ pos_plane
    return depth, contact_point_plane[:3], contact_point_sphere[:3]


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
    vd = pos1 - pos0  # direction vector from pos0 to pos1
    dist = np.linalg.norm(vd)  # distance between the two particles
    if dist >= radius0 + radius1:
        return None
    vd /= dist
    # calculate the penetration depth
    depth = dist - radius0 - radius1
    # calculate the contact point on the first particle
    contact_point_0 = pos0 + radius0 * vd
    # calculate the contact point on the second particle
    contact_point_1 = pos1 - radius1 * vd
    return depth, contact_point_0, contact_point_1


def collide_shape_shape(
    shape1: fcl.CollisionObject,
    shape2: fcl.CollisionObject,
    geom1: fcl.CollisionGeometry,
    id1: int,
    id2: int,
    ctype: ContactType,
    contacts: Contacts,  # OUTPUT: to store the contacts
) -> None:
    """
    Detect collisions between two shapes.
    """
    request = fcl.CollisionRequest()
    request.enable_contact = True
    request.num_max_contacts = 100
    result = fcl.CollisionResult()
    ret = fcl.collide(shape1, shape2, request, result)
    if ret == 0:
        return
    # print(f"result.contacts={len(result.contacts)}")
    for c in result.contacts:
        contacts.contact_instance0.append(id1)
        contacts.contact_instance1.append(id2)
        contacts.contact_type.append(ctype)
        contacts.contact_point0.append(c.pos)
        contacts.contact_point1.append(c.pos)
        # Normal is pointing from o1 to o2,
        # if id(c.o1) == id(geom1), normal is from geom1 to geom2
        # otherwise, normal is from geom2 to geom1
        contacts.contact_normal.append(c.normal if id(c.o1) == id(geom1) else -c.normal)
        contacts.contact_depth.append(c.penetration_depth)


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
        - earliest collision time: float
        - contact normal: nparray (3,) in world frame when the earliest collision happens.
        None if no collision is detected.
    """
    vel0 = pos0_out - pos0_in
    vel1 = pos1_out - pos1_in
    vel_rel = vel1 - vel0  # velocity of particle 1 w.r.t. particle 0
    pos_rel = pos1_in - pos0_in
    # polynomial for the distance between the two particles
    a = np.dot(vel_rel, vel_rel)
    if a < 1e-12:
        # parallel velocity. Not an approaching contact.
        return None

    b = 2.0 * np.dot(vel_rel, pos_rel)
    c = np.dot(pos_rel, pos_rel) - (radius0 + radius1) ** 2
    if b * b - 4 * a * c < 0.0:
        # This is just for numerical stability. Sometimes b^2-4ac is slightly negative due to
        # floating point errors, which corresponds to no collision
        return None
    rs = np.roots([a, b, c])
    if rs.size < 2:
        return None
    t0 = min(rs[0], rs[1])
    t1 = max(rs[0], rs[1])
    if t1 < 0.0 or t0 > 1.0:
        return None
    # now t0 <= 1 and t1 >= 0
    if t0 < 0.0:
        # now t1 > 0, otherwise we would have returned None earlier.
        # check t = 0
        if b < 0.0:
            return 0.0, pos_rel / np.linalg.norm(pos_rel)
        else:
            return None
    else:
        # 0.0 <= t0 <= 1.0
        # assert not np.iscomplex(t0), f"t0={t0}, t1={t1}"
        dd = pos_rel + t0 * vel_rel
        return t0, dd / np.linalg.norm(dd)


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
                            contacts.contact_normal.append(model.shape_transform[i, :3, 1])
                            contacts.contact_depth.append(-depth)

        # Detect contacts between particles.
        for i in range(1, model.particle_count):
            for j in range(i):
                ret = collide_particle_particle(
                    state.particle_q[i],
                    state.particle_q[j],
                    model.particle_radius[i],
                    model.particle_radius[j],
                )
                if ret is not None:
                    depth, contact_point_0, contact_point_1 = ret
                    if depth < 0.0:
                        contacts.contact_instance0.append(i)
                        contacts.contact_instance1.append(j)
                        contacts.contact_type.append(ContactType.PARTICLE_PARTICLE)
                        contacts.contact_point0.append(contact_point_0)
                        contacts.contact_point1.append(contact_point_1)
                        nrm = contact_point_0 - state.particle_q[i]
                        nrm /= np.linalg.norm(nrm)  # SAFE. nrm will not be zero.
                        contacts.contact_normal.append(nrm)
                        contacts.contact_depth.append(-depth)

        # Detect mesh-mesh and mesh-plane contacts.
        model.update_collision_objects(state)
        for i in range(model.shape_count):
            for j in range(i):
                if (
                    model.shape_flags[i] & ShapeFlags.ACTIVE.value == 0
                    and model.shape_flags[j] & ShapeFlags.ACTIVE.value == 0
                ):
                    continue
                if model.shape_flags[i] & ShapeFlags.ACTIVE.value == 0:
                    collide_shape_shape(
                        model.shape_collision_objects[i],
                        model.shape_collision_objects[j],
                        model.shape_fcl_geoms[i],
                        i,
                        j,
                        ContactType.FIXED_SHAPE_SHAPE,
                        contacts,
                    )
                elif model.shape_flags[j] & ShapeFlags.ACTIVE.value == 0:
                    collide_shape_shape(
                        model.shape_collision_objects[j],
                        model.shape_collision_objects[i],
                        model.shape_fcl_geoms[j],
                        j,
                        i,
                        ContactType.FIXED_SHAPE_SHAPE,
                        contacts,
                    )
                else:
                    collide_shape_shape(
                        model.shape_collision_objects[i],
                        model.shape_collision_objects[j],
                        model.shape_fcl_geoms[i],
                        i,
                        j,
                        ContactType.SHAPE_SHAPE,
                        contacts,
                    )
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
                    ret = continuous_collide_particle_plane(
                        model.shape_transform[i],
                        model.shape_inv_transform[i],
                        pos0=state_0.particle_q[j],  # starting position
                        pos1=state_1.particle_q[j],  # ending position
                        radius=model.particle_radius[j],
                    )
                    if ret is not None:
                        contacts.contact_instance0_continuous.append(i)
                        contacts.contact_instance1_continuous.append(j)
                        contacts.contact_type_continuous.append(ContactType.FIXED_SHAPE_PARTICLE)
                        contacts.contact_normal_continuous.append(ret)

        # Detect contacts between particles.
        for i in range(1, model.particle_count):
            for j in range(i):
                ret = continuous_collide_particle_particle(
                    state_0.particle_q[i],
                    state_1.particle_q[i],
                    state_0.particle_q[j],
                    state_1.particle_q[j],
                    model.particle_radius[i],
                    model.particle_radius[j],
                )
                if ret is not None:
                    contacts.contact_instance0_continuous.append(i)
                    contacts.contact_instance1_continuous.append(j)
                    contacts.contact_type_continuous.append(ContactType.PARTICLE_PARTICLE)
                    contacts.contact_normal_continuous.append(ret[1])
