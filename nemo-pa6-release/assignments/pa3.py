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

import yaml
from rich import print as rprint

from assignments.plot import PlotSpec
from nemo.geometry.types import ParticleFlags
from nemo.sim import Model, ModelBuilder
from nemo.solvers import (
    ContactPenaltySolver,
    MidpointSolver,
    MultiContactSolver,
    SolverBase,
    SymplecticEulerSolver,
)


def load_scene(config: str) -> tuple[Model, SolverBase, PlotSpec | None]:
    with open(config) as file:
        config_data = yaml.safe_load(file)

    sconfig = config_data["solver"]
    if "gravity" in sconfig:
        gravity = sconfig["gravity"]
    else:
        gravity = -9.81

    builder = ModelBuilder(gravity=gravity)
    if "particles" in config_data:
        # add particles
        for p in config_data["particles"]:
            # position
            pos = [float(v) for v in p["pos"]]
            # velocity
            if "vel" in p:
                vel = [float(v) for v in p["vel"]]
            else:
                vel = [0.0, 0.0, 0.0]
            radius = float(p["radius"]) if "radius" in p else None
            drag = float(p["drag"]) if "drag" in p else None
            restitution = float(p["restitution"]) if "restitution" in p else None
            flags = 0 if p.get("fixed") else ParticleFlags.ACTIVE.value
            builder.add_particle(
                pos, vel, p.get("mass"), radius=radius, drag=drag, flags=flags, restitution_coeff=restitution
            )

    if builder.particle_count == 0:
        raise RuntimeError("No particles are added. Can't run simulation")

    if "springs" in config_data:
        # load springs
        for s in config_data["springs"]:
            ps = s["particle_ids"]
            if len(ps) != 2:
                raise RuntimeError(f"particle_ids must be two index numbers, not {ps}")
            builder.add_spring(ps[0], ps[1], s["stiffness"], s.get("damping"), s.get("rest_length"))

    if "gravitational" in config_data:
        # load gravitational pairs
        for g in config_data["gravitational"]:
            ps = g["particle_ids"]
            if len(ps) != 2:
                raise RuntimeError(f"particle_ids must be two index numbers, not {ps}")
            builder.add_gravitational(ps[0], ps[1], g["G"])

    if "fixed_shapes" in config_data:
        # Load fixed shapes
        for s in config_data["fixed_shapes"]:
            shape_t = s["type"]
            if shape_t == "half_space_plane":
                stiffness = s["penalty_stiffness"] if "penalty_stiffness" in s else None
                damping = s["penalty_damping"] if "penalty_damping" in s else None
                builder.add_shape_half_space_plane(
                    plane_normal=s["normal"],
                    plane_offset=s["offset"],
                    stiffness=stiffness,
                    damping=damping,
                    restitution_coeff=s["restitution"] if "restitution" in s else None,
                )
            else:
                raise RuntimeError(f"Unknown shape type: [{shape_t}]")

    model = builder.finalize()
    rprint("[bold green]Loading scene ...")
    rprint(f"  {model.particle_count} particles are added")
    rprint(f"  {model.spring_count} springs are added")
    rprint(f"  {model.gravitational_count} gravitational pairs are added")
    rprint(f"  {model.shape_count} fixed shapes are added")

    if sconfig["integrator"].lower() == "symplectic":
        integrator = SymplecticEulerSolver(model, sconfig["timestep"])
    elif sconfig["integrator"].lower() == "midpoint":
        integrator = MidpointSolver(model, sconfig["timestep"])
    else:
        raise RuntimeError(f"Unknown integrator type: [{sconfig['integrator']}]")

    if sconfig["type"].lower() == "contact_penalty":
        solver = ContactPenaltySolver(
            model,
            sconfig["timestep"],
            stiffness=sconfig["particle_contact_penalty_stiffness"]
            if "particle_contact_penalty_stiffness" in sconfig
            else None,
            damping=sconfig["particle_contact_penalty_damping"]
            if "particle_contact_penalty_damping" in sconfig
            else None,
            integrator=integrator,
        )
    elif sconfig["type"].lower() == "multi_contact_iterative":
        solver = MultiContactSolver(
            model,
            sconfig["timestep"],  # dt
            # maxits
            maxits=sconfig["max_its"] if "max_its" in sconfig else None,
            # stiffness
            stiffness=sconfig["particle_contact_penalty_stiffness"]
            if "particle_contact_penalty_stiffness" in sconfig
            else None,
            # damping
            damping=sconfig["particle_contact_penalty_damping"]
            if "particle_contact_penalty_damping" in sconfig
            else None,
            # iterator
            integrator=integrator,
        )
    else:
        raise RuntimeError(f"Unknown solver type: [{sconfig['type']}]")

    if "plot" in config_data:
        plot = PlotSpec(config_data["plot"])
        if plot.particle_id >= model.particle_count:
            raise RuntimeError(f"Particle ID{plot.particle_id} is out of range")
        if plot.dof < 0 or plot.dof > 2:
            raise RuntimeError(f"DoF{plot.dof} is out of range, must be 0, 1, or 2")
        if plot.y_range_min >= plot.y_range_max:
            raise RuntimeError(f"Y_range [{plot.y_range_min}, {plot.y_range_max}] is not valid")
        return model, solver, plot
    else:
        return model, solver, None
