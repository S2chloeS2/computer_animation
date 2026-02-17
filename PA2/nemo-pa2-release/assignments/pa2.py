import yaml
from rich import print as rprint

from assignments.plot import PlotSpec
from nemo.geometry import ParticleFlags
from nemo.sim import Model, ModelBuilder
from nemo.solvers import (
    ExplicitEulerSolver,
    ImplicitEulerSolver,
    LinearizedImplicitSolver,
    MidpointSolver,
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
            flags = 0 if p.get("fixed") else ParticleFlags.ACTIVE.value
            builder.add_particle(pos, vel, p.get("mass"), radius=radius, drag=drag, flags=flags)

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

    model = builder.finalize()
    rprint("[bold green]Loading scene ...")
    rprint(f"  {model.particle_count} particles are added")
    rprint(f"  {model.spring_count} springs are added")
    rprint(f"  {model.gravitational_count} gravitational pairs are added")

    if sconfig["type"].lower() == "explicit_euler":
        solver = ExplicitEulerSolver(model, sconfig["timestep"])
    elif sconfig["type"].lower() == "symplectic_euler":
        solver = SymplecticEulerSolver(model, sconfig["timestep"])
    elif sconfig["type"].lower() == "midpoint":
        solver = MidpointSolver(model, sconfig["timestep"])
    elif sconfig["type"].lower() == "linearized_implicit":
        solver = LinearizedImplicitSolver(model, sconfig["timestep"])
    elif sconfig["type"].lower() == "implicit_euler":
        solver = ImplicitEulerSolver(model, sconfig["timestep"])
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
