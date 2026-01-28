import yaml
from rich import print as rprint

from nemo.geometry import ParticleFlags
from nemo.sim import Model, ModelBuilder
from nemo.solvers import ExplicitEulerSolver, SolverBase


def load_scene(config: str) -> tuple[Model, SolverBase]:
    with open(config) as file:
        config_data = yaml.safe_load(file)

    builder = ModelBuilder()
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
            flags = 0 if p.get("fixed") else ParticleFlags.ACTIVE.value
            builder.add_particle(pos, vel, p.get("mass"), radius, flags)

    if "springs" in config_data:
        # load springs
        for s in config_data["springs"]:
            ps = s["particle_ids"]
            if len(ps) != 2:
                raise RuntimeError(f"particle_ids must be two index numbers, not {ps}")
            builder.add_spring(ps[0], ps[1], s["stiffness"], s.get("damping"), s.get("rest_length"))

    if builder.particle_count == 0:
        raise RuntimeError("No particles are added. Can't run simulation")

    rprint("[bold green]Loading scene ...")
    rprint(f"  {builder.particle_count} particles are added")

    model = builder.finalize()
    sconfig = config_data["solver"]
    if sconfig["type"].lower() == "explicit_euler":
        solver = ExplicitEulerSolver(model, sconfig["timestep"])
    else:
        raise RuntimeError(f"Unknown solver type: [{sconfig['type']}]")
    return model, solver
