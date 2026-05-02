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

from pathlib import Path

import trimesh
import yaml
from rich import print as rprint

from assignments.plot import PlotSpec
from nemo.sim import Model, ModelBuilder
from nemo.solvers import ContactPenaltySolver, LCPGSSolver, SolverBase, SymplecticEulerSolver


def load_scene(config: str) -> tuple[Model, SolverBase, PlotSpec | None]:
    with open(config) as file:
        config_data = yaml.safe_load(file)

    sconfig = config_data["solver"]
    if "gravity" in sconfig:
        gravity = sconfig["gravity"]
    else:
        gravity = -9.81

    builder = ModelBuilder(gravity=gravity)

    if "shapes" in config_data:
        for s in config_data["shapes"]:
            shape_t = s["type"]
            if shape_t == "sphere":
                stiffness = s["penalty_stiffness"] if "penalty_stiffness" in s else None
                damping = s["penalty_damping"] if "penalty_damping" in s else None
                builder.add_shape_sphere(
                    pos=s["pos"],
                    radius=s["radius"],
                    stiffness=stiffness,
                    damping=damping,
                    restitution_coeff=s["restitution"] if "restitution" in s else None,
                    density=s["density"] if "density" in s else 1.0,
                    fixed=s["fixed"] if "fixed" in s else False,
                )
            elif shape_t == "cylinder":
                stiffness = s["penalty_stiffness"] if "penalty_stiffness" in s else None
                damping = s["penalty_damping"] if "penalty_damping" in s else None
                orient = s["orient"] if "orient" in s else [1.0, 0.0, 0.0, 0.0]
                builder.add_shape_cylinder(
                    pos=s["pos"],
                    radius=s["radius"],
                    lz=s["length"],
                    orient=orient,
                    stiffness=stiffness,
                    damping=damping,
                    restitution_coeff=s["restitution"] if "restitution" in s else None,
                    density=s["density"] if "density" in s else 1.0,
                    fixed=s["fixed"] if "fixed" in s else False,
                )
            elif shape_t == "box":
                stiffness = s["penalty_stiffness"] if "penalty_stiffness" in s else None
                damping = s["penalty_damping"] if "penalty_damping" in s else None
                orient = s["orient"] if "orient" in s else [1.0, 0.0, 0.0, 0.0]
                builder.add_shape_box(
                    pos=s["pos"],
                    size=s["size"],
                    orient=orient,
                    stiffness=stiffness,
                    damping=damping,
                    restitution_coeff=s["restitution"] if "restitution" in s else None,
                    density=s["density"] if "density" in s else 1.0,
                    fixed=s["fixed"] if "fixed" in s else False,
                )
            elif shape_t == "mesh":
                mfile = Path(s["file"])
                if not mfile.is_absolute():
                    mfile = Path(config).parent / mfile
                mesh = trimesh.load(mfile)
                if "scale" in s:
                    mesh.vertices *= s["scale"]
                builder.add_shape_mesh(
                    mesh=mesh,
                    pos=s["pos"],
                    orient=s["orient"] if "orient" in s else [1.0, 0.0, 0.0, 0.0],
                    stiffness=s["penalty_stiffness"] if "penalty_stiffness" in s else None,
                    damping=s["penalty_damping"] if "penalty_damping" in s else None,
                    restitution_coeff=s["restitution"] if "restitution" in s else None,
                    density=s["density"] if "density" in s else 1.0,
                    fixed=s["fixed"] if "fixed" in s else False,
                )
            elif shape_t == "half_space_plane":
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
    rprint(f"  {model.shape_count} shapes are added")

    if sconfig["integrator"].lower() == "symplectic":
        integrator = SymplecticEulerSolver(model, sconfig["timestep"])
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
    elif sconfig["type"].lower() == "lcp_gs":
        solver = LCPGSSolver(
            model,
            sconfig["timestep"],
            integrator=integrator,
        )
    else:
        raise RuntimeError(f"Unknown solver type: [{sconfig['type']}]")

    if "plot" in config_data:
        plot = PlotSpec(config_data["plot"])
        if plot.shape_id >= model.shape_count:
            raise RuntimeError(f"Shape ID{plot.shape_id} is out of range")
        if plot.dof < 0 or plot.dof > 6:
            raise RuntimeError(f"DoF{plot.dof} is out of range, must be in [0, 6]")
        if plot.y_range_min >= plot.y_range_max:
            raise RuntimeError(f"Y_range [{plot.y_range_min}, {plot.y_range_max}] is not valid")
        if plot.shape_id >= 0:
            rprint(f"  Plotting body #{plot.shape_id} with DoF #{plot.dof}")
        return model, solver, plot
    else:
        return model, solver, None
