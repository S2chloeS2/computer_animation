#
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Written by Changxi Zheng <cxz@cs.columbia.edu>, 2026
#
import importlib
from collections.abc import Callable
from typing import Annotated

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import polyscope.implot as psplot
import trimesh
import typer
from rich import print as rprint

import nemo
from nemo.core import Axis, header
from nemo.geometry.types import GeoType, ParticleFlags, ShapeFlags
from nemo.sim import Model, State, body_q_to_transform
from nemo.solvers import SolverBase

from .plot import PlotSpec

app = typer.Typer(add_completion=False)
FPS = 60


class Runner_PA_1_2:
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

    runner = Runner_PA_1_2(model, solver, callback=data_accum_callback if plspec is not None else None)

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
                    psplot.PlotLine("Vertical Pos.", np.asarray(data_x), np.asarray(data_z))
                    psplot.EndPlot()

    ps.set_user_callback(callback)
    rprint("[bold green]Launch simulation ...")
    runner.launch()


# ------------------------------------------------------------------------------------------------


class Runner_PA_3:
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

        ps.set_ground_plane_mode("none")
        # register particles in the viewer
        self.particle_views = []
        r = 0.0
        for ii in range(model.particle_count):
            color = None
            # label fixed particles with Red
            if model.particle_flags[ii] & ParticleFlags.ACTIVE.value == 0:
                color = (1.0, 0.0, 0.0)
            pc = ps.register_point_cloud(
                f"partile-{ii:06d}",
                self.state_0.particle_q[ii : ii + 1, :],
                color=color,
            )
            pc.set_radius(self.model.particle_radius[ii], relative=False)
            self.particle_views.append(pc)
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
        # Add plane
        for i in range(self.model.shape_count):
            if self.model.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                plane = ps.add_slice_plane()
                plane.set_pose(
                    plane_position=self.model.shape_transform[i, :3, 3],
                    plane_normal=self.model.shape_transform[i, :3, 1],
                )
                plane.set_draw_plane(True)
                plane.set_transparency(1.0)
                plane.set_draw_widget(False)

    def check(self):
        """Sanity check the scene configuration."""
        contacts = self.solver.collision.instantaneous_contacts(self.state_0)
        if len(contacts.contact_instance0) > 0:
            raise RuntimeError("Contacts detected in the initial state")

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


def launch_pa_3(model: Model, solver: SolverBase, plspec: PlotSpec | None = None):
    data_x = []
    data_z = []

    def data_accum_callback(ts: float, state: State):
        data_x.append(ts)
        data_z.append(state.particle_q[plspec.particle_id, plspec.dof])
        if len(data_x) > 1000:
            data_x.pop(0)
            data_z.pop(0)

    runner = Runner_PA_3(model, solver, callback=data_accum_callback if plspec is not None else None)
    runner.check()

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

        if plspec is not None:
            plot_len = len(data_x)
            if plot_len > 5:
                if psplot.BeginPlot("Vertical Position", flags=psplot.ImPlotAxisFlags_AutoFit):
                    psplot.SetupAxisLimits(psplot.ImAxis_X1, data_x[0], data_x[-1], psplot.ImPlotCond_Always)
                    psplot.SetupAxisLimits(
                        psplot.ImAxis_Y1, plspec.y_range_min, plspec.y_range_max, psplot.ImPlotCond_Always
                    )
                    psplot.PlotLine("Vertical Pos.", np.asarray(data_x), np.asarray(data_z))
                    psplot.EndPlot()

    ps.set_user_callback(callback)
    rprint("[bold green]Launch simulation ...")
    runner.launch()


# ------------------------------------------------------------------------------------------------


class Runner_PA_4:
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

        ps.look_at([0.0, -2.0, 2.0], [0.0, 0.0, 0.0])
        ps.set_ground_plane_mode("none")
        self.shape_views = []
        # Add plane
        plane_mesh = trimesh.creation.box(extents=(100.0, 0.01, 100.0))
        cylinder_mesh = None
        box_mesh = None

        for i in range(self.model.shape_count):
            if self.model.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                plane = ps.register_surface_mesh(f"PLANE{i:06d}", plane_mesh.vertices, plane_mesh.faces, material="wax")
                plane.set_transform(self.model.shape_transform[i])
                self.shape_views.append(plane)
            elif self.model.shape_type[i] == GeoType.SPHERE:
                if self.model.shape_flags[i] & ShapeFlags.ACTIVE.value == 0:
                    color = (1.0, 0.0, 0.0)
                else:
                    color = None
                pc = ps.register_point_cloud(
                    f"SPHERE{i:06d}",
                    self.state_0.shape_q[i : i + 1, :3],
                    color=color,
                )
                pc.set_radius(self.model.shape_fcl_geoms[i].radius, relative=False)
                self.shape_views.append(pc)
            elif self.model.shape_type[i] == GeoType.CYLINDER:
                if cylinder_mesh is None:
                    cylinder_mesh = trimesh.creation.cylinder(
                        radius=self.model.shape_fcl_geoms[i].radius, height=self.model.shape_fcl_geoms[i].lz
                    )
                pc = ps.register_surface_mesh(
                    f"CYLINDER{i:06d}", cylinder_mesh.vertices, cylinder_mesh.faces, material="wax"
                )
                pc.set_transform(self.model.shape_transform[i])
                self.shape_views.append(pc)
            elif self.model.shape_type[i] == GeoType.BOX:
                if box_mesh is None:
                    box_mesh = trimesh.creation.box(extents=self.model.shape_fcl_geoms[i].side)
                pc = ps.register_surface_mesh(f"BOX{i:06d}", box_mesh.vertices, box_mesh.faces, material="wax")
                pc.set_transform(self.model.shape_transform[i])
                self.shape_views.append(pc)
            elif self.model.shape_type[i] == GeoType.MESH:
                pc = ps.register_surface_mesh(
                    f"MESH{i:06d}",
                    self.model.shape_meshes[i].vertices,
                    self.model.shape_meshes[i].faces,
                    material="wax",
                )
                pc.set_edge_color((0.0, 0.0, 0.0))
                pc.set_transform(self.model.shape_transform[i])
                self.shape_views.append(pc)
            else:
                raise RuntimeError(f"Unknown shape type: {self.model.shape_type[i]}")

    def check(self):
        """Sanity check the scene configuration."""
        contacts = self.solver.collision.instantaneous_contacts(self.state_0)
        if len(contacts.contact_instance0) > 0:
            raise RuntimeError("Contacts detected in the initial state")

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
                    self.solver.step(self.state_0, self.state_1)
                    # update particle positions
                    self.state_0, self.state_1 = self.state_1, self.state_0

                # update the viewer states for rendering
                for i in range(self.model.shape_count):
                    if self.model.shape_type[i] == GeoType.HALF_SPACE_PLANE:
                        continue
                    if self.model.shape_type[i] == GeoType.SPHERE:
                        self.shape_views[i].update_point_positions(self.state_0.shape_q[i : i + 1, :3])
                    else:
                        self.shape_views[i].set_transform(body_q_to_transform(self.state_0.shape_q[i]))

                if self.callback is not None:
                    self.callback(self.solver.ts, self.state_0)

            ps.frame_tick()  # renders one UI frame, returns immediately
            if self.screenshot:
                ps.screenshot()


def launch_pa_4(model: Model, solver: SolverBase, plspec: PlotSpec | None = None):
    data_x = []
    data_z = []

    def data_accum_callback(ts: float, state: State):
        data_x.append(ts)
        data_z.append(state.shape_q[plspec.shape_id, plspec.dof])
        if len(data_x) > 1000:
            data_x.pop(0)
            data_z.pop(0)

    runner = Runner_PA_4(model, solver, callback=data_accum_callback if plspec is not None else None)
    runner.check()
    wireframe = False

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

        nonlocal wireframe
        changed, wireframe = psim.Checkbox("Wireframe", wireframe)
        if changed:
            for i in range(model.shape_count):
                if model.shape_type[i] == GeoType.MESH:
                    if wireframe:
                        runner.shape_views[i].set_edge_width(1.0)
                    else:
                        runner.shape_views[i].set_edge_width(0.0)

        if plspec is not None:
            plot_len = len(data_x)
            if plot_len > 5:
                if psplot.BeginPlot("Vertical Position", flags=psplot.ImPlotAxisFlags_AutoFit):
                    psplot.SetupAxisLimits(psplot.ImAxis_X1, data_x[0], data_x[-1], psplot.ImPlotCond_Always)
                    psplot.SetupAxisLimits(
                        psplot.ImAxis_Y1, plspec.y_range_min, plspec.y_range_max, psplot.ImPlotCond_Always
                    )
                    psplot.PlotLine("Vertical Pos.", np.asarray(data_x), np.asarray(data_z))
                    psplot.EndPlot()

    ps.set_user_callback(callback)
    rprint("[bold green]Launch simulation ...")
    runner.launch()


# ------------------------------------------------------------------------------------------------


class Runner_PA_5:
    def __init__(self, model: Model, solver: SolverBase):
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

        ps.look_at([0.0, -2.0, 2.0], [0.0, 0.0, 0.0])
        # ps.set_ground_plane_mode("none")
        ps.set_ground_plane_height(0.0)

        if model.tri_count > 0:
            self.cloth_front = ps.register_surface_mesh(
                "cloth_front",
                self.state_0.particle_q,
                model.tri_indices,
            )
            self.cloth_front.set_color((0.2, 0.5, 1.0))
            self.cloth_front.set_back_face_policy("cull")
            self.cloth_back = ps.register_surface_mesh(
                "cloth_back",
                self.state_0.particle_q,
                model.tri_indices[:, ::-1],
            )
            self.cloth_back.set_color((1.0, 0.3, 0.2))
            self.cloth_back.set_back_face_policy("cull")

    def launch(self):
        """Launch the interactive simulation with a GUI"""
        header.show()
        rprint("[bold green]---------------------------------------------------------------------------")
        print("Press [space] to toggle start/pause of the simulation")

        while not ps.window_requests_close():
            # timestep the simulation
            if self.running:
                self.solver.step(self.state_0, self.state_1)
                # update particle positions
                self.state_0, self.state_1 = self.state_1, self.state_0

                # update the viewer states for rendering
                self.cloth_front.update_vertex_positions(self.state_0.particle_q)
                self.cloth_back.update_vertex_positions(self.state_0.particle_q)

            ps.frame_tick()  # renders one UI frame, returns immediately
            if self.screenshot:
                ps.screenshot()


def launch_pa_5(model: Model, solver: SolverBase, plspec: PlotSpec | None = None):
    runner = Runner_PA_5(model, solver)
    wireframe = False

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

        nonlocal wireframe
        changed, wireframe = psim.Checkbox("Wireframe", wireframe)
        if changed:
            for i in range(model.shape_count):
                if model.shape_type[i] == GeoType.MESH:
                    if wireframe:
                        runner.shape_views[i].set_edge_width(1.0)
                    else:
                        runner.shape_views[i].set_edge_width(0.0)

    ps.set_user_callback(callback)
    rprint("[bold green]Launch simulation ...")
    runner.launch()


# ------------------------------------------------------------------------------------------------


# Entry point for PA1
@app.command("pa1", help="Run PA1 simulation")
def pa1(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa1", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_1_2(model, solver, plspec)


# Entry point for PA2
@app.command("pa2", help="Run PA2 simulation")
def pa2(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa2", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_1_2(model, solver, plspec)


# Entry point for PA3
@app.command("pa3", help="Run PA3 simulation")
def pa3(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa3", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_3(model, solver, plspec)


# Entry point for PA4
@app.command("pa4", help="Run PA4 simulation")
def pa4(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa4", package=__package__)
    # 1. Load configuration and create model
    model, solver, plspec = module.load_scene(config)
    launch_pa_4(model, solver, plspec)


# Entry point for PA4
@app.command("pa5", help="Run PA5 simulation")
def pa5(config: Annotated[str, typer.Argument(help="Scene configuration file")]):
    module = importlib.import_module(".pa5", package=__package__)
    # 1. Load configuration and create model
    model, solver = module.load_scene(config)
    launch_pa_5(model, solver)


if __name__ == "__main__":
    app()
