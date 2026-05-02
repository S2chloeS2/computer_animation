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

import numpy as np
import yaml
from rich import print as rprint

from nemo.sim import Model
from nemo.solvers import FluidFlipSolver, SolverBase


def load_scene(config: str) -> tuple[Model, SolverBase]:
    with open(config) as file:
        config_data = yaml.safe_load(file)
    if "fluid" in config_data:
        s = config_data["fluid"]
        fluid_cfile = Path(s["initial_file"])
        if not fluid_cfile.is_absolute():
            fluid_cfile = Path(config).parent / fluid_cfile
        cell_size = s["cell_size"]
        fluid_map = []
        l0 = 0
        with open(fluid_cfile) as fs:
            for lnum, line in enumerate(fs):
                ll = line.strip()
                if len(ll) == 0:
                    continue
                if l0 > 0:
                    if len(ll) != l0:
                        raise RuntimeError(f"Line {lnum} has length {len(ll)}, expected {l0}")
                else:
                    l0 = len(ll)
                fluid_map.append(ll)
    else:
        raise RuntimeError("No fluid domain is specified in the config file")

    model = Model()
    model.init_fluid_map = fluid_map
    model.fluid_cell_size = cell_size
    model.fluid_domain_res = np.array([len(fluid_map[0]), len(fluid_map)], dtype=np.uint32)
    model.fluid_domain_size = model.fluid_domain_res * cell_size

    sconfig = config_data["solver"]
    if sconfig["type"].lower() == "fluid_flip":
        solver = FluidFlipSolver(model, sconfig["timestep"])
    else:
        raise RuntimeError(f"Unsupported solver type: [{sconfig['type']}]")

    rprint("[bold green]Loading scene ...[/bold green]")
    rprint(f"  [yellow]{model.fluid_domain_res[0]}x{model.fluid_domain_res[1]}[/yellow] fluid domain is added")
    rprint(f"  {model.fluid_cell_size} cell size is used")

    return model, solver
