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

from nemo.sim import Model, ModelBuilder
from nemo.solvers import LargeStepClothSolver, SolverBase


def load_scene(config: str) -> tuple[Model, SolverBase]:
    with open(config) as file:
        config_data = yaml.safe_load(file)

    sconfig = config_data["solver"]
    if "gravity" in sconfig:
        gravity = sconfig["gravity"]
    else:
        gravity = -9.81

    def extract_params(shape: dict, name: str):
        if name in shape:
            r = shape[name]
            return r[0], r[1]
        return None, None

    builder = ModelBuilder(gravity=gravity)
    if "shapes" in config_data:
        for s in config_data["shapes"]:
            shape_t = s["type"]
            if shape_t == "cloth":
                mfile = Path(s["file"])
                if not mfile.is_absolute():
                    mfile = Path(config).parent / mfile
                mesh = trimesh.load(mfile)
                if "scale" in s:
                    mesh.vertices *= s["scale"]
                ks, ksd = extract_params(s, "stretch_params")
                kr, krd = extract_params(s, "shear_params")
                kb, kbd = extract_params(s, "bending_params")
                builder.add_cloth_mesh(
                    mesh,
                    pos=s["pos"],
                    orient=s["orient"] if "orient" in s else [1.0, 0.0, 0.0, 0.0],
                    density=s["density"] if "density" in s else 1.0,
                    ks=ks,
                    ksd=ksd,
                    kr=kr,
                    krd=krd,
                    kb=kb,
                    kbd=kbd,
                    fixed=s["fixed"] if "fixed" in s else None,
                )
            else:
                raise RuntimeError(f"Unsupported shape type: [{shape_t}]")

    model = builder.finalize()
    rprint("[bold green]Loading scene ...")
    rprint(f"  {model.tri_count} cloth triangles are added")
    rprint(f"  {model.edge_count} cloth edges are added")

    solver_type = sconfig["type"].lower()
    if solver_type == "large_step_cloth":
        solver = LargeStepClothSolver(
            model,
            sconfig["timestep"],
        )
    else:
        raise RuntimeError(f"Unsupported solver type: [{sconfig['type']}]")

    return model, solver
