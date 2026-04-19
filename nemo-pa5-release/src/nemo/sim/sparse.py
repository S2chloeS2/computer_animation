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

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..core.types import nparray, override
from ..geometry.types import ParticleFlags
from ..sim.model import Model


def min_max(a: int, b: int) -> tuple[int, int]:
    if a < b:
        return a, b
    else:
        return b, a


class ParticleIndexCache:
    """This is a slight variant of the CSR format indicating nonzero elements
    in the upper triangular part of the matrix. The first N elements are the diagonal blocks.

    N: number of 3x3 blocks in one dimension
    """

    def __init__(self, model: Model):
        N = model.particle_count

        self.N = N
        self.index_map = np.full((N, N), -1, dtype=np.int32)
        for i in range(N):
            self.index_map[i, i] = i

        cnt = N

        def label_nonzero(a: int, b: int):
            nonlocal cnt
            if (
                model.particle_flags[a] & ParticleFlags.ACTIVE.value != 0
                and model.particle_flags[b] & ParticleFlags.ACTIVE.value != 0
                and self.index_map[a, b] == -1
            ):
                self.index_map[a, b] = 0  # upper triangular part
                cnt += 1

        if model.tri_count > 0:
            for t0, t1, t2 in model.tri_indices:
                label_nonzero(*min_max(t0, t1))  # a < b
                label_nonzero(*min_max(t1, t2))
                label_nonzero(*min_max(t0, t2))
        self.nonzero_count = cnt

        self.col_indices = np.empty(cnt, dtype=np.int32)
        self.row_ptr = np.empty((N, 2), dtype=np.int32)
        """Row pointer for the sparse matrix.
        row_ptr[i, 0] = start index of nonzero elements in the i-th row
        row_ptr[i, 1] = end index of nonzero elements in the i-th row
        """
        cnt = N
        for i in range(N):
            self.row_ptr[i, 0] = cnt
            for j in range(i + 1, N):
                if self.index_map[i, j] != -1:
                    self.index_map[i, j] = cnt
                    self.col_indices[cnt] = j
                    cnt += 1
            self.row_ptr[i, 1] = cnt
        assert cnt == self.nonzero_count

        self.mass_mat = np.empty((N, 3, 3), dtype=np.float64)
        for i in range(N):
            if model.particle_flags[i] & ParticleFlags.ACTIVE.value != 0:  # active particle
                self.mass_mat[i] = np.eye(3) * model.particle_mass[i]
            else:
                self.mass_mat[i] = np.eye(3)


class SparseParticleSystem(LinearOperator):
    """A sparse particle system.
    Given N particles. This class represents a 3N x 3N sparse matrix.
    The matrix can be viewed as NxN blocks, each block is a 3x3 matrix, emerged from an
    the implicit Euler time integrator (see Course Notes of PA2).

    The class is a linear operator, so it can be used with `scipy.sparse.linalg.solve`
    or `scipy.sparse.linalg.cg`.
    """

    def __init__(self, model: Model, index_cache: ParticleIndexCache):
        """
        At construction time, the system is initialized as a block diagonal matrix `M`
        """
        n = index_cache.N * 3
        super().__init__(dtype=np.float64, shape=(n, n))
        self.index_cache = index_cache
        self.ele_mat = np.zeros((index_cache.nonzero_count, 3, 3), dtype=np.float64)
        self.ele_mat[: index_cache.N] = index_cache.mass_mat

    @property
    def block_size(self) -> int:
        return self.index_cache.N

    @property
    def matrix_size(self) -> int:
        return self.index_cache.N * 3

    @override
    def _matvec(self, x):
        assert x.shape[0] == self.matrix_size

        ret = np.zeros_like(x)
        for i in range(self.block_size):
            # diagonal block
            ret[i * 3 : (i + 1) * 3] += self.ele_mat[i] @ x[i * 3 : (i + 1) * 3]
            # off-diagonal blocks
            for j in range(self.index_cache.row_ptr[i, 0], self.index_cache.row_ptr[i, 1]):
                col_idx = self.index_cache.col_indices[j]
                assert col_idx > i
                # block at [i, col_idx]
                ret[i * 3 : (i + 1) * 3] += self.ele_mat[j] @ x[col_idx * 3 : (col_idx + 1) * 3]
                # this is also a block at [col_idx, i]
                ret[col_idx * 3 : (col_idx + 1) * 3] += self.ele_mat[j].T @ x[i * 3 : (i + 1) * 3]
        return ret

    @override
    def _rmatvec(self, x):
        """
        compute A^H @ x
        """
        return self._matvec(x)

    def accumu_block(self, i: int, j: int, blk: nparray, s: float = 1.0) -> None:
        """
        Accumulate a 3x3 block into the [i, j] block of the sparse matrix.

        Because this is a symmetric matrix, we only need to accumulate the upper triangular part.
        """
        assert i <= j
        idx = self.index_cache.index_map[i, j]
        if blk.shape != (3, 3):
            print(f"blk.shape: {blk.shape}")
            print(f"blk: {blk}")
            print(f"i: {i}, j: {j}")
            print(f"idx: {idx}")
            print(f"s: {s}")
        assert idx >= 0 and blk.shape == (3, 3)
        self.ele_mat[idx] += s * blk


class JacobiParticlePrecond(LinearOperator):
    """A Jacobi preconditioner for the sparse particle system.

    Given a sparse particle system A, this class represents a diagonal matrix D,
    where D[i, i] = 1 / A[i, i].
    """

    def __init__(self, A: SparseParticleSystem):
        super().__init__(dtype=np.float64, shape=(A.matrix_size, A.matrix_size))
        self._diag = np.empty((A.block_size, 3, 3), dtype=np.float64)
        # first N elements of A are diagonal blocks
        for i in range(A.block_size):
            self._diag[i] = np.linalg.inv(A.ele_mat[i])

    @override
    def _rmatvec(self, x):
        """
        compute P^H @ x
        """
        return self._matvec(x)

    @override
    def _matvec(self, x):
        """
        compute P @ x
        """
        ret = self._diag @ x.reshape(-1, 3)[..., np.newaxis]
        return ret.reshape(-1)
