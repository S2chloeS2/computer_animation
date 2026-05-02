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

from enum import IntEnum

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rich import print as rprint
from scipy.ndimage import convolve

from ..core.types import nparray
from ..sim.model import Model
from ..sim.state import State
from .solver import SolverBase


class VoxelType(IntEnum):
    """Type of voxel."""

    AIR = 0
    """Air voxel."""
    WATER = 1
    """Water voxel."""


def extrapolate_velocity_field(u: nparray, valid: nparray) -> nparray:
    """Extrapolate the velocity field by one voxel layer.
    Air voxels may produce zero velocity on voxel edges. When transferring velocity from grid to particles,
    the particles may need the velocities stored in those air voxel edges. As a result, the interpolated particle
    velocities will be incorrectly slowed down by the air voxel edges.

    This implmentation utilize scipy's convolve function for efficiency.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)

    neighbor_sum = convolve(u * valid, kernel, mode="constant", cval=0)
    neighbor_count = convolve(valid.astype(float), kernel, mode="constant", cval=0)

    u_extrapolated = u.copy()
    fill_mask = (~valid) & (neighbor_count > 0)
    u_extrapolated[fill_mask] = neighbor_sum[fill_mask] / neighbor_count[fill_mask]
    return u_extrapolated


class FluidFlipSolver(SolverBase):
    """This solver implement the fluic FLIP solver as described in the lecture.

    NOTE: This solver considers only the fluid domain in the model and ignores the rest of the model
    (such as shapes, particles, etc.).
    """

    def __init__(self, model: Model, dt: float):
        super().__init__(model=model, dt=dt)

        # check if the model contains a fluid domain
        if model.fluid_domain_res is None:
            raise ValueError("This solver expects a fluid domain in the model.")
        self._voxel2index: nparray = np.empty((model.fluid_domain_res[1], model.fluid_domain_res[0]), dtype=np.uint32)
        """ Map each voxel to its index in a 1D array, this is the order used by the pressure vector
        in Poisson solve
        """
        # total number of voxels
        sz = model.fluid_domain_res[0] * model.fluid_domain_res[1]
        self._index2voxel: nparray = np.empty((sz, 2), dtype=np.int32)
        """ Map an index in linear system dimension to a voxel coordinate
        (inverse mapping of _voxel2index)
        """
        self._n_water_voxels = 0
        # number of water voxels (varying in time)

        self._GRAVITY = -9.81
        self._FLIP_FACTOR = 0.95
        """FLIP weights to blend FLIP and PIC results. If this factor = 1, a complete FLIP method is used.
        (see course notes for details)
        """

        # NOTE: here I left the data structures that I used in my implementation as a reference. 
        # Feel free to change/refactor the data structures as you see fit.
        self._voxel_type: nparray = np.empty((model.fluid_domain_res[1], model.fluid_domain_res[0]), dtype=np.uint8)
        self._u_x_weights = np.empty((model.fluid_domain_res[1], model.fluid_domain_res[0] + 1), dtype=np.float64)
        self._u_y_weights = np.empty((model.fluid_domain_res[1] + 1, model.fluid_domain_res[0]), dtype=np.float64)

        # RHS of the pressure solve linear system
        self._pressure_rhs = np.empty(sz, dtype=np.float64)
        self._last_pressure = np.empty((model.fluid_domain_res[1], model.fluid_domain_res[0]), dtype=np.float64)
        # To store the pressure value at the last timestep
        self._p0 = np.empty(sz, dtype=np.float64)
        # To store the pressure initialization
        # LHS of the pressure solve linear system (sparse matrix)
        self._pressure_A_rows = np.empty(sz, dtype=np.int32)  # row indices of nonzero elements
        self._pressure_A_cols = np.empty(sz, dtype=np.int32)  # column indices of nonzero elements
        self._pressure_A_vals = np.empty(sz, dtype=np.float64)  # values of nonzero elements

        self._valid_u_x = None
        self._valid_u_y = None

    def _transfer_particle_velocity_to_grid(self, par_voxel_coord: nparray, state: State):
        """
        Transfer the particle velocity to the grid.

        Args:
            par_voxel_coord: nparray, shape (particle_count, 2): the voxel coordinates of the particles
            state: State: the state of the simulation
        """
        model = self.model
        assert state.fluid_u_y is not None and state.fluid_u_x is not None
        assert state.fluid_particle_qd is not None and state.fluid_particle_q is not None
        assert state.fluid_u_x.shape == self._u_x_weights.shape and state.fluid_u_y.shape == self._u_y_weights.shape

        self._u_x_weights.fill(0)
        self._u_y_weights.fill(0)
        # initialize grid velocity to 0
        state.fluid_u_x.fill(0)
        state.fluid_u_y.fill(0)

        dx = model.fluid_cell_size
        res_y, res_x = model.fluid_domain_res[1], model.fluid_domain_res[0]
        n_particles = state.fluid_particle_q.shape[0]

        for p in range(n_particles):
            px, py = state.fluid_particle_q[p, 0], state.fluid_particle_q[p, 1]
            vx, vy = state.fluid_particle_qd[p, 0], state.fluid_particle_qd[p, 1]
            wx, wy = px / dx, py / dx

            # --- u_x: staggered at (ix*dx, (iy+0.5)*dx), so offset = (wx, wy-0.5) ---
            fx, fy = wx, wy - 0.5
            i0, j0 = int(np.floor(fx)), int(np.floor(fy))
            i1, j1 = i0 + 1, j0 + 1
            s, t = fx - i0, fy - j0
            for (jj, ii, w) in [(j0, i0, (1-s)*(1-t)), (j0, i1, s*(1-t)),
                                 (j1, i0, (1-s)*t),    (j1, i1, s*t)]:
                if 0 <= jj < res_y and 0 <= ii <= res_x:
                    state.fluid_u_x[jj, ii] += w * vx
                    self._u_x_weights[jj, ii] += w

            # --- u_y: staggered at ((ix+0.5)*dx, iy*dx), so offset = (wx-0.5, wy) ---
            fx, fy = wx - 0.5, wy
            i0, j0 = int(np.floor(fx)), int(np.floor(fy))
            i1, j1 = i0 + 1, j0 + 1
            s, t = fx - i0, fy - j0
            for (jj, ii, w) in [(j0, i0, (1-s)*(1-t)), (j0, i1, s*(1-t)),
                                 (j1, i0, (1-s)*t),    (j1, i1, s*t)]:
                if 0 <= jj <= res_y and 0 <= ii < res_x:
                    state.fluid_u_y[jj, ii] += w * vy
                    self._u_y_weights[jj, ii] += w

        # normalize by accumulated weights
        mask_x = self._u_x_weights > 0
        state.fluid_u_x[mask_x] /= self._u_x_weights[mask_x]
        mask_y = self._u_y_weights > 0
        state.fluid_u_y[mask_y] /= self._u_y_weights[mask_y]

        # track which edges have valid (water-influenced) velocity
        self._valid_u_x = mask_x
        self._valid_u_y = mask_y

    def _voxel_out_of_domain(self, ix: int, iy: int) -> bool:
        """Check if a voxel coordinate is out of the fluid domain."""
        return ix < 0 or ix >= self.model.fluid_domain_res[0] or iy < 0 or iy >= self.model.fluid_domain_res[1]

    def _pressure_projection(self, state: State) -> None:
        # construct linear system Ax = b
        dx = self.model.fluid_cell_size
        # count no. non-zero elements in the sparse matrix
        nnz = 0
        for i in range(self._n_water_voxels):
            ix = self._index2voxel[i, 0]  # x-index of the voxel
            iy = self._index2voxel[i, 1]  # y-index of the voxel
            for nx, ny in [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]:
                if not self._voxel_out_of_domain(nx, ny) and self._voxel_type[ny, nx] == VoxelType.WATER.value:
                    nnz += 1
            nnz += 1
        # Allocate memory for the sparse matrix
        if self._pressure_A_rows.shape[0] < nnz:
            self._pressure_A_rows = np.empty(nnz, dtype=np.int32)
            self._pressure_A_cols = np.empty(nnz, dtype=np.int32)
            self._pressure_A_vals = np.empty(nnz, dtype=np.float64)


        # Construct the non-zero elements of the pressure solve linear system.
        dt = self.dt
        k = 0
        for i in range(self._n_water_voxels):
            ix = self._index2voxel[i, 0]
            iy = self._index2voxel[i, 1]
            diag = 0
            for nx, ny in [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]:
                if self._voxel_out_of_domain(nx, ny):
                    # SOLID wall: no flux, no contribution
                    pass
                elif self._voxel_type[ny, nx] == VoxelType.WATER:
                    # WATER neighbor: off-diagonal -1
                    j = self._voxel2index[ny, nx]
                    self._pressure_A_rows[k] = i
                    self._pressure_A_cols[k] = j
                    self._pressure_A_vals[k] = -1.0
                    k += 1
                    diag += 1
                else:
                    # AIR neighbor: pressure=0 free surface → increases diagonal
                    diag += 1
            # diagonal entry
            self._pressure_A_rows[k] = i
            self._pressure_A_cols[k] = i
            self._pressure_A_vals[k] = float(diag)
            k += 1

            # RHS: -divergence * (dx / dt)
            div = (state.fluid_u_x[iy, ix + 1] - state.fluid_u_x[iy, ix]
                   + state.fluid_u_y[iy + 1, ix] - state.fluid_u_y[iy, ix])
            self._pressure_rhs[i] = -(dx / dt) * div

        if self._n_water_voxels == 0:
            return

        A = sp.csc_matrix(
            (self._pressure_A_vals[:k], (self._pressure_A_rows[:k], self._pressure_A_cols[:k])),
            shape=(self._n_water_voxels, self._n_water_voxels),
        )

        # warm-start from last frame's pressure
        self._p0[: self._n_water_voxels] = 0.0
        for i in range(self._n_water_voxels):
            ix, iy = self._index2voxel[i, 0], self._index2voxel[i, 1]
            self._p0[i] = self._last_pressure[iy, ix]

        ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=5)
        M = spla.LinearOperator(A.shape, ilu.solve)
        p, info = spla.cg(
            A,
            self._pressure_rhs[: self._n_water_voxels],
            M=M,
            maxiter=100,
            x0=self._p0[: self._n_water_voxels],
        )
        if info > 0:
            rprint(f"[red]Warning:[/red] Pressure solve not converge (its = {info})")
        elif info < 0:
            rprint("[red]Error:[/red] Pressure solve failed")

        # cache pressure for next frame warm-start
        self._last_pressure.fill(0.0)
        for i in range(self._n_water_voxels):
            ix, iy = self._index2voxel[i, 0], self._index2voxel[i, 1]
            self._last_pressure[iy, ix] = p[i]

        # correct grid velocities with pressure gradient: u -= (dt/dx) * grad(p)
        scale = dt / dx
        res_x = int(self.model.fluid_domain_res[0])
        res_y = int(self.model.fluid_domain_res[1])

        # u_x[iy, ix]: between cell (ix-1, iy) and (ix, iy)
        for ix in range(1, res_x):
            for iy in range(res_y):
                left  = self._voxel_type[iy, ix - 1] == VoxelType.WATER
                right = self._voxel_type[iy, ix]     == VoxelType.WATER
                if left or right:
                    p_right = p[self._voxel2index[iy, ix]]     if right else 0.0
                    p_left  = p[self._voxel2index[iy, ix - 1]] if left  else 0.0
                    state.fluid_u_x[iy, ix] -= scale * (p_right - p_left)

        # u_y[iy, ix]: between cell (ix, iy-1) and (ix, iy)
        for ix in range(res_x):
            for iy in range(1, res_y):
                below = self._voxel_type[iy - 1, ix] == VoxelType.WATER
                above = self._voxel_type[iy,     ix] == VoxelType.WATER
                if below or above:
                    p_above = p[self._voxel2index[iy,     ix]] if above else 0.0
                    p_below = p[self._voxel2index[iy - 1, ix]] if below else 0.0
                    state.fluid_u_y[iy, ix] -= scale * (p_above - p_below)

    def _transfer_grid_velocity_to_particles(self, par_voxel_coord: nparray, state_in: State, state_out: State):
        model = self.model

        dx = model.fluid_cell_size

        # extrapolate grid velocities into air voxels to avoid zero-velocity contamination
        u_x_now = extrapolate_velocity_field(state_out.fluid_u_x, self._valid_u_x)
        u_y_now = extrapolate_velocity_field(state_out.fluid_u_y, self._valid_u_y)
        u_x_old = extrapolate_velocity_field(state_in.fluid_u_x, self._valid_u_x)
        u_y_old = extrapolate_velocity_field(state_in.fluid_u_y, self._valid_u_y)

        n_particles = state_out.fluid_particle_q.shape[0]
        for p in range(n_particles):
            px, py = state_out.fluid_particle_q[p, 0], state_out.fluid_particle_q[p, 1]
            wx, wy = px / dx, py / dx

            # --- interpolate u_x at (wx, wy-0.5) ---
            fx, fy = wx, wy - 0.5
            i0, j0 = int(np.floor(fx)), int(np.floor(fy))
            i1, j1 = i0 + 1, j0 + 1
            s, t = fx - i0, fy - j0
            res_y_x = u_x_now.shape[0]
            res_x_x = u_x_now.shape[1]

            def sample(grid, jj, ii):
                jj = max(0, min(jj, grid.shape[0] - 1))
                ii = max(0, min(ii, grid.shape[1] - 1))
                return grid[jj, ii]

            u_x_pic  = ((1-s)*(1-t)*sample(u_x_now, j0, i0) + s*(1-t)*sample(u_x_now, j0, i1)
                      + (1-s)*t    *sample(u_x_now, j1, i0) + s*t    *sample(u_x_now, j1, i1))
            u_x_old_i = ((1-s)*(1-t)*sample(u_x_old, j0, i0) + s*(1-t)*sample(u_x_old, j0, i1)
                       + (1-s)*t    *sample(u_x_old, j1, i0) + s*t    *sample(u_x_old, j1, i1))

            # --- interpolate u_y at (wx-0.5, wy) ---
            fx, fy = wx - 0.5, wy
            i0, j0 = int(np.floor(fx)), int(np.floor(fy))
            i1, j1 = i0 + 1, j0 + 1
            s, t = fx - i0, fy - j0

            u_y_pic  = ((1-s)*(1-t)*sample(u_y_now, j0, i0) + s*(1-t)*sample(u_y_now, j0, i1)
                      + (1-s)*t    *sample(u_y_now, j1, i0) + s*t    *sample(u_y_now, j1, i1))
            u_y_old_i = ((1-s)*(1-t)*sample(u_y_old, j0, i0) + s*(1-t)*sample(u_y_old, j0, i1)
                       + (1-s)*t    *sample(u_y_old, j1, i0) + s*t    *sample(u_y_old, j1, i1))

            # FLIP: particle old velocity + grid velocity change
            u_x_flip = state_in.fluid_particle_qd[p, 0] + (u_x_pic - u_x_old_i)
            u_y_flip = state_in.fluid_particle_qd[p, 1] + (u_y_pic - u_y_old_i)

            # PIC/FLIP blend
            state_out.fluid_particle_qd[p, 0] = u_x_pic * (1.0 - self._FLIP_FACTOR) + u_x_flip * self._FLIP_FACTOR
            state_out.fluid_particle_qd[p, 1] = u_y_pic * (1.0 - self._FLIP_FACTOR) + u_y_flip * self._FLIP_FACTOR

    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """Step the solver forward in time."""
        if dt is None:
            dt = self.dt
        self.ts += dt

        model = self.model
        assert state_in.fluid_particle_qd is not None and state_in.fluid_particle_q is not None
        assert state_out.fluid_u_x is not None and state_out.fluid_u_y is not None

        dx = model.fluid_cell_size
        res_x, res_y = int(model.fluid_domain_res[0]), int(model.fluid_domain_res[1])
        domain_w = model.fluid_domain_size[0]
        domain_h = model.fluid_domain_size[1]

        # 1. advect particles: q_new = q_old + v * dt, clamp to domain
        state_out.fluid_particle_q = state_in.fluid_particle_q + state_in.fluid_particle_qd * dt
        state_out.fluid_particle_q[:, 0] = np.clip(state_out.fluid_particle_q[:, 0], 0.0, domain_w - 1e-6)
        state_out.fluid_particle_q[:, 1] = np.clip(state_out.fluid_particle_q[:, 1], 0.0, domain_h - 1e-6)
        state_out.fluid_particle_qd = state_in.fluid_particle_qd.copy()

        # voxel coordinates of each particle
        par_voxel_coord = (state_out.fluid_particle_q / dx).astype(np.int32)
        par_voxel_coord[:, 0] = np.clip(par_voxel_coord[:, 0], 0, res_x - 1)
        par_voxel_coord[:, 1] = np.clip(par_voxel_coord[:, 1], 0, res_y - 1)

        # 2. transfer particle velocity to grid
        self._transfer_particle_velocity_to_grid(par_voxel_coord, state_out)

        # 2.1 classify voxels as WATER or AIR, build index mappings for pressure solve
        self._voxel_type.fill(VoxelType.AIR)
        self._voxel_type[par_voxel_coord[:, 1], par_voxel_coord[:, 0]] = VoxelType.WATER
        idx = 0
        for iy in range(res_y):
            for ix in range(res_x):
                if self._voxel_type[iy, ix] == VoxelType.WATER:
                    self._voxel2index[iy, ix] = idx
                    self._index2voxel[idx] = [ix, iy]
                    idx += 1
        self._n_water_voxels = idx

        # 2.2 copy grid velocity into state_in for FLIP (old velocity reference)
        state_in.fluid_u_x = state_out.fluid_u_x.copy()
        state_in.fluid_u_y = state_out.fluid_u_y.copy()

        # 3. apply gravity to u_y edges adjacent to at least one WATER voxel
        for ix in range(res_x):
            for iy in range(1, res_y):
                if (self._voxel_type[iy - 1, ix] == VoxelType.WATER or
                        self._voxel_type[iy, ix] == VoxelType.WATER):
                    state_out.fluid_u_y[iy, ix] += self._GRAVITY * dt

        # 4. pressure projection (Step 3)
        self._pressure_projection(state_out)

        # 5. grid-to-particle transfer with PIC/FLIP blend (Step 4)
        self._transfer_grid_velocity_to_particles(par_voxel_coord, state_in, state_out)
