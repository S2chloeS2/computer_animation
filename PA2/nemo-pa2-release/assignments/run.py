import importlib
from collections.abc import Callable
from typing import Annotated

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import polyscope.implot as psplot
import typer
from rich import print as rprint

import nemo
from nemo.core import Axis, header
from nemo.geometry import ParticleFlags
from nemo.sim import Model, State
from nemo.solvers import SolverBase

from .plot import PlotSpec

app = typer.Typer(add_completion=False)
FPS = 60


class Runner:
    def __init__(self, model: Model, solver: SolverBase, callback: Callable | None = None):
        """
        Args:
            callback: Callable
        """
        self.model = model
        self.solver = solver
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.running = False
        self.screenshot = False
        self.callback = callback

        # Set up viewer
        ps.set_program_name(f"Nemo {nemo.__version__}")
        ps.set_build_default_gui_panels(False)
        ps.set_give_focus_on_show(True)
        ps.set_frame_tick_limit_fps_mode("block_to_hit_target")
        # ps.set_frame_tick_limit_fps_mode("skip_frames_to_hit_target")
        ps.set_max_fps(FPS)
        ps.init()
        if model.up_axis == Axis.Y:
            ps.set_up_dir("y_up")
        elif model.up_axis == Axis.Z:
            ps.set_up_dir("z_up")
        elif model.up_axis == Axis.X:
            ps.set_up_dir("x_up")

        ps.set_ground_plane_height(0.0)
        # register particles in the viewer
        self.particle_views = []
        r = 0.0
        for ii in range(model.particle_count):
            color = None
            # label fixed particles with Red
            if model.particle_flags[ii] & ParticleFlags.ACTIVE.value == 0:
                color = (1.0, 0.0, 0.0)
            self.particle_views.append(
                ps.register_point_cloud(
                    f"partile-{ii:06d}",
                    self.state_1.particle_q[ii : ii + 1, :],
                    radius=self.model.particle_radius[ii],
                    color=color,
                )
            )
            r += self.model.particle_radius[ii]
        r /= model.particle_count
        # register springs in the viewer
        self.spring_view = None
        if self.model.spring_count > 0:
            self.spring_view = ps.register_curve_network(
                "springs",
                self.state_1.particle_q,
                self.model.spring_indices,
                radius=r * 0.3,
                color=(0.988, 0.678, 0.008),
            )

    def launch(self):
        """Launch the interactive simulation with a GUI"""
        header.show()
        rprint("[bold green]---------------------------------------------------------------------------")
        print("Press [space] to toggle start/pause of the simulation")

        ts = 0.0
        dt_render = 1.0 / FPS
        while not ps.window_requests_close():
            # timestep the simulation
            if self.running:
                ts += dt_render

                # advance the simulation state until it caches the
                # render time. In this way, the displayed animation is
                # agnotic to the simulation timestep (i.e., reducing timestep size
                # wouldn't slow down the displayed animation). No matter what timestep size
                # you choose (as long as it's not too small or too large), the viewer displays
                # the simulated progress in realtime.
                while self.solver.ts < ts:
                    self.state_0.clear_forces()
                    self.solver.step(self.state_0, self.state_1)
                    # update particle positions
                    self.state_0, self.state_1 = self.state_1, self.state_0

                # update the viewer states for rendering
                for ii in range(self.model.particle_count):
                    self.particle_views[ii].update_point_positions(self.state_0.particle_q[ii : ii + 1, :])
                if self.spring_view is not None:
                    self.spring_view.update_node_positions(self.state_0.particle_q)
                if self.callback is not None:
                    self.callback(self.solver.ts, self.state_0)

            ps.frame_tick()  # renders one UI frame, returns immediately
            if self.screenshot:
                ps.screenshot()


# ------------------------------------------------------------------------------------------------


def launch_pa_1_2(model: Model, solver: SolverBase, plspec: PlotSpec | None = None):
    data_x = []
    data_z = []

    def data_accum_callback(ts: float, state: State):
        data_x.append(ts)
        data_z.append(state.particle_q[plspec.particle_id, plspec.dof])
        if len(data_x) > 1000:
            data_x.pop(0)
            data_z.pop(0)

    runner = Runner(model, solver, callback=data_accum_callback if plspec is not None else None)

    show_ground = True

    def callback():
        # ps.build_structure_gui()
        # io = psim.GetIO()
        # if io.MouseClicked[0]:
        #     print("HERE 1")
        if psim.IsKeyReleased(psim.ImGuiKey_Space):
            runner.running = not runner.running
        if psim.IsKeyReleased(psim.ImGuiKey_R):
            runner.screenshot = not runner.screenshot
        if runner.running:
            psim.Text(f"Simulation is RUNNING (t={solver.ts:03f}s)")
        else:
            psim.Text("Simulation is Paused")

        # toggle ground plane
        nonlocal show_ground
        changed, show_ground = psim.Checkbox("Show Ground", show_ground)
        if changed:
            if show_ground:
                ps.set_ground_plane_mode("tile_reflection")
            else:
                ps.set_ground_plane_mode("none")

        if plspec is not None:
            plot_len = len(data_x)
            if plot_len > 5:
                if psplot.BeginPlot("Vertical Position", flags=psplot.ImPlotAxisFlags_AutoFit):
                    psplot.SetupAxisLimits(psplot.ImAxis_X1, data_x[0], data_x[-1], psplot.ImPlotCond_Always)
                    psplot.SetupAxisLimits(
                        psplot.ImAxis_Y1, plspec.y_range_min, plspec.y_range_max, psplot.ImPlotCond_Always
                    )
                    xs = np.asarray(data_x, dtype=np.float64)[:plot_len]
                    ys = np.asarray(data_z, dtype=np.float64)[:plot_len]
                    psplot.PlotLine("Vertical Pos.", xs, ys)
                    psplot.EndPlot()

    ps.set_user_callback(callback)
    rprint("[bold green]Launch simulation ...")
    runner.launch()


# Entry point for PA1
@app.command("pa1", help="Run PA1 simulation")
def pa1(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa1", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_1_2(model, solver, plspec)


# Entry point for PA1
@app.command("pa2", help="Run PA2 simulation")
def pa2(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa2", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_1_2(model, solver, plspec)


if __name__ == "__main__":
    app()
