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
        # TODO: iterate over all particles and transfer their velocity to the grid

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
        # ...
        
        # After collecting the non-zero elements, you need to construct the sparse matrix and solve the linear system.
        # Here I simply use scipy's sparse matrix construction to construct the linear system.
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

        # A = sp.csc_matrix(
        #     (self._pressure_A_vals[:nnz], (self._pressure_A_rows[:nnz], self._pressure_A_cols[:nnz])),
        #     shape=(self._n_water_voxels, self._n_water_voxels),
        # )

        # Here I use incomplete LU (ILU) preconditioner to speed up the conjugate gradient (CG) solve.
        # ILU preconditioner
        # ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=5)
        # M = spla.LinearOperator(A.shape, ilu.solve)
        # # solve the linear system
        # p, info = spla.cg(
        #     A,
        #     self._pressure_rhs[: self._n_water_voxels],
        #     M=M,
        #     maxiter=100,
        #     x0=self._p0[: self._n_water_voxels],
        #     # callback=callback,
        # )
        # if info > 0:
        #     rprint(f"[red]Warning:[/red] Pressure solve not converge (its = {info})")
        # elif info < 0:
        #     rprint("[red]Error:[/red] Pressure solve failed")

        # After the pressure solve, correct grid velocity due to pressure gradient
        # You also cache the pressure values by updating the _last_pressure array.
        # The cached pressure values can be used for warm start the next pressure solve.
        # (see x0 argument above in spla.cg)

    def _transfer_grid_velocity_to_particles(self, par_voxel_coord: nparray, state_in: State, state_out: State):
        model = self.model

        # Here comes the extrapolation that I talked about in the lecture.
        # This extrapolation is to avoid using undefined velocities for grid-to-particle interpolation.
        # Again, following is the code that I used in my implementation as a reference.
        #
        # u_x_now = extrapolate_velocity_field(state_out.fluid_u_x, self._valid_u_x)
        # u_y_now = extrapolate_velocity_field(state_out.fluid_u_y, self._valid_u_y)
        # u_x_old = extrapolate_velocity_field(state_in.fluid_u_x, self._valid_u_x)
        # u_y_old = extrapolate_velocity_field(state_in.fluid_u_y, self._valid_u_y)

        # TODO: Now transfer the extrapolated grid velocity to particles

        # As a hint, the final particle velocity is a weighted blend of the PIC and FLIP velocities.
        #
        # state_out.fluid_particle_qd[i, 0] = u_x_pic * (1.0 - self._FLIP_FACTOR) + u_x_flip * self._FLIP_FACTOR
        # state_out.fluid_particle_qd[i, 1] = u_y_pic * (1.0 - self._FLIP_FACTOR) + u_y_flip * self._FLIP_FACTOR

    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """Step the solver forward in time."""
        if dt is None:
            dt = self.dt
        self.ts += dt

        model = self.model
        assert state_in.fluid_particle_qd is not None and state_in.fluid_particle_q is not None
        assert state_out.fluid_u_x is not None and state_out.fluid_u_y is not None

        # TODO: Implement your FLIP solver algorithm here.

        # Here are the steps you need to follow:
        # 1. advect particles
        # NOTE: after advection, make sure all particles stays within the simulation domain

        # 2. transfer particle velocity to grid
        # As a hint, I implmented this transfer in a separate function:
        #   _transfer_particle_velocity_to_grid(par_voxel_coord, state_out)
        # 2.1 classify all voxels as fluid voxel or air voxel
        # 2.2 make a copy to prepare for FLIP
        # 3.apply gravity to grid velocity

        # 4. solve the pressure on grid
        # As a hint, I implmented this pressure projection in a separate function:
        # self._pressure_projection(state_out)

        # 5. transfer grid velocity to particles
        # As a hint, I implmented this transfer in a separate function:
        # self._transfer_grid_velocity_to_particles(par_voxel_coord, state_in, state_out)

        # For all those helper functions, I left them there for you as a
        # reference. Feel free to change/refactor them as you see fit.
