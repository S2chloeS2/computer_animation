# ----------------------------------------------------------------------------
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# PA5 self-check script.
#
# HOW TO USE:
#   1. Paste your PA5 implementations of eval_cloth_bending_forces and
#      eval_cloth_stretch_shear_forces into the two stubs below.
#   2. From the repository root, run:
#          python -m assignments.pa5_check
#   3. Each check prints [PASS] or [FAIL] with error magnitudes. Three passes
#      means your implementations match the reference on scene04.yml.
#
# The harness loads scene04.yml, re-uses a seeded random-perturbed state
# committed in the .npz fixture, calls your pasted functions, and compares
# outputs to the pre-computed ground truth. Do not modify code below the
# "TEST HARNESS" banner.
# ----------------------------------------------------------------------------
import hashlib
import sys
from pathlib import Path

import numpy as np

from nemo.geometry.types import ParticleFlags
from nemo.sim.sparse import ParticleIndexCache, SparseParticleSystem

from .pa5 import load_scene


# =============================================================
# PASTE YOUR IMPLEMENTATION BELOW
# =============================================================


def _cloth_wuv(model, state, tri):
    t0, t1, t2 = model.tri_indices[tri]
    return (
        np.column_stack(
            (state.particle_q[t1] - state.particle_q[t0], state.particle_q[t2] - state.particle_q[t0]),
        )
        @ model.tri_uv_inv[tri]
    )


def eval_cloth_bending_forces(model, state):
    for ii in range(model.edge_count):
        v0, v1, v2, v3 = model.edge_indices[ii]
        kb, kbd = model.edge_materials[ii]
        theta_rest = model.edge_rest_angle[ii]
        x1, x2, x3, x4 = state.particle_q[v0], state.particle_q[v1], state.particle_q[v2], state.particle_q[v3]
        e = x2 - x1
        en = np.linalg.norm(e)
        n1 = np.cross(e, x3 - x1)
        n2 = np.cross(x4 - x1, e)
        n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
        if en < 1e-10 or n1n < 1e-10 or n2n < 1e-10:
            continue
        cos_t = np.clip(np.dot(n1, n2) / (n1n * n2n), -1.0, 1.0)
        sin_t = np.dot(np.cross(n1 / n1n, n2 / n2n), e / en)
        C = np.arctan2(sin_t, cos_t) - theta_rest
        g3 = -(en / n1n ** 2) * n1
        g4 = -(en / n2n ** 2) * n2
        g1 = -((x2 - x3) @ e / en ** 2) * g3 - ((x2 - x4) @ e / en ** 2) * g4
        g2 =  ((x1 - x3) @ e / en ** 2) * g3 + ((x1 - x4) @ e / en ** 2) * g4
        bend_verts = [(v0, g1), (v1, g2), (v2, g3), (v3, g4)]
        kbC = kb * C
        for vi, gi in bend_verts:
            state.particle_f[vi] -= kbC * gi
        C_dot = sum(np.dot(gi, state.particle_qd[vi]) for vi, gi in bend_verts)
        for vi, gi in bend_verts:
            state.particle_f[vi] -= kbd * C_dot * gi


def eval_cloth_stretch_shear_forces(model, state, A, h):
    h2 = h * h
    for tri in range(model.tri_count):
        t0, t1, t2 = model.tri_indices[tri]
        ks, ksd, kr, krd = model.tri_materials[tri]
        wuv = _cloth_wuv(model, state, tri)
        a = model.tri_areas[tri]
        d = model.tri_uv_inv[tri]
        w_u, w_v = wuv[:, 0], wuv[:, 1]

        d_u = [-(d[0,0]+d[1,0]), d[0,0], d[1,0]]
        d_v = [-(d[0,1]+d[1,1]), d[0,1], d[1,1]]
        I3d = np.eye(3)

        t0a = model.particle_flags[t0] & ParticleFlags.ACTIVE.value != 0
        t1a = model.particle_flags[t1] & ParticleFlags.ACTIVE.value != 0
        t2a = model.particle_flags[t2] & ParticleFlags.ACTIVE.value != 0

        # Shear
        C_s = np.dot(w_u, w_v)
        gs0 = -(d[0,0]+d[1,0])*w_v - (d[0,1]+d[1,1])*w_u
        gs1 =   d[0,0]*w_v + d[0,1]*w_u
        gs2 =   d[1,0]*w_v + d[1,1]*w_u
        sc = a * kr * C_s
        if t0a: state.particle_f[t0] -= sc * gs0
        if t1a: state.particle_f[t1] -= sc * gs1
        if t2a: state.particle_f[t2] -= sc * gs2
        verts = [(t0, gs0, t0a), (t1, gs1, t1a), (t2, gs2, t2a)]
        h2akr = h2 * a * kr
        for ii, (ti, gi, ai) in enumerate(verts):
            for jj, (tj, gj, aj) in enumerate(verts[ii:], start=ii):
                if ai and aj:
                    i_, j_ = (ti, tj) if ti <= tj else (tj, ti)
                    gi_, gj_ = (gi, gj) if ti <= tj else (gj, gi)
                    coeff2 = d_u[ii]*d_v[jj] + d_v[ii]*d_u[jj]
                    blk = np.outer(gi_, gj_) + C_s * coeff2 * I3d
                    A.accumu_block(i_, j_, h2akr * blk)

        # Stretch
        wu_n, wv_n = np.linalg.norm(w_u), np.linalg.norm(w_v)
        if wu_n > 1e-10 and wv_n > 1e-10:
            wu_hat, wv_hat = w_u/wu_n, w_v/wv_n
            C_u, C_v = wu_n - 1.0, wv_n - 1.0
            gu0 = -(d[0,0]+d[1,0])*wu_hat; gu1 = d[0,0]*wu_hat; gu2 = d[1,0]*wu_hat
            gv0 = -(d[0,1]+d[1,1])*wv_hat; gv1 = d[0,1]*wv_hat; gv2 = d[1,1]*wv_hat
            aks = a * ks
            if t0a: state.particle_f[t0] -= aks*(C_u*gu0 + C_v*gv0)
            if t1a: state.particle_f[t1] -= aks*(C_u*gu1 + C_v*gv1)
            if t2a: state.particle_f[t2] -= aks*(C_u*gu2 + C_v*gv2)
            sv = [(t0,gu0,gv0,t0a),(t1,gu1,gv1,t1a),(t2,gu2,gv2,t2a)]
            h2aks = h2 * aks
            I_wu = I3d - np.outer(wu_hat, wu_hat)
            I_wv = I3d - np.outer(wv_hat, wv_hat)
            for ii,(ti,gui,gvi,ai) in enumerate(sv):
                for jj,(tj,guj,gvj,aj) in enumerate(sv[ii:], start=ii):
                    if ai and aj:
                        i_,j_ = (ti,tj) if ti<=tj else (tj,ti)
                        blk = (np.outer(gui,guj)+np.outer(gvi,gvj)) if ti<=tj else (np.outer(guj,gui)+np.outer(gvj,gvi))
                        coeff_u2 = d_u[ii]*d_u[jj]/wu_n
                        coeff_v2 = d_v[ii]*d_v[jj]/wv_n
                        blk += C_u*coeff_u2*I_wu + C_v*coeff_v2*I_wv
                        A.accumu_block(i_,j_, h2aks*blk)
            # Stretch damping
            v0q,v1q,v2q = state.particle_qd[t0],state.particle_qd[t1],state.particle_qd[t2]
            Cud = np.dot(gu0,v0q)+np.dot(gu1,v1q)+np.dot(gu2,v2q)
            Cvd = np.dot(gv0,v0q)+np.dot(gv1,v1q)+np.dot(gv2,v2q)
            aksd = a*ksd
            if t0a: state.particle_f[t0] -= aksd*(Cud*gu0+Cvd*gv0)
            if t1a: state.particle_f[t1] -= aksd*(Cud*gu1+Cvd*gv1)
            if t2a: state.particle_f[t2] -= aksd*(Cud*gu2+Cvd*gv2)
            haksd = h*aksd
            for ii,(ti,gui,gvi,ai) in enumerate(sv):
                for jj,(tj,guj,gvj,aj) in enumerate(sv[ii:], start=ii):
                    if ai and aj:
                        i_,j_ = (ti,tj) if ti<=tj else (tj,ti)
                        blk = (np.outer(gui,guj)+np.outer(gvi,gvj)) if ti<=tj else (np.outer(guj,gui)+np.outer(gvj,gvi))
                        A.accumu_block(i_,j_, haksd*blk)

        # Shear damping
        v0q,v1q,v2q = state.particle_qd[t0],state.particle_qd[t1],state.particle_qd[t2]
        Csd = np.dot(gs0,v0q)+np.dot(gs1,v1q)+np.dot(gs2,v2q)
        akrd = a*krd
        if t0a: state.particle_f[t0] -= akrd*Csd*gs0
        if t1a: state.particle_f[t1] -= akrd*Csd*gs1
        if t2a: state.particle_f[t2] -= akrd*Csd*gs2
        hakrd = h*akrd
        for ii,(ti,gi,ai) in enumerate(verts):
            for jj,(tj,gj,aj) in enumerate(verts[ii:], start=ii):
                if ai and aj:
                    i_,j_ = (ti,tj) if ti<=tj else (tj,ti)
                    gi_,gj_ = (gi,gj) if ti<=tj else (gj,gi)
                    A.accumu_block(i_,j_, hakrd*np.outer(gi_,gj_))


# =============================================================
# TEST HARNESS (do not modify)
# =============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "scenes" / "pa5" / "scene04.yml"
GT_PATH = REPO_ROOT / "src" / "tests" / "data" / "pa5_scene04_ground_truth.npz"
ATOL, RTOL = 1e-8, 1e-6


def _hash_index_map(idx_map: np.ndarray) -> int:
    return int(hashlib.md5(idx_map.tobytes()).hexdigest()[:15], 16)


def _state_from_gt(model, gt):
    state = model.state()
    np.copyto(state.particle_q, gt["particle_q"])
    np.copyto(state.particle_qd, gt["particle_qd"])
    state.clear_forces()
    return state


def _report_array(name, got, expected):
    diff = got - expected
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    if np.allclose(got, expected, atol=ATOL, rtol=RTOL):
        print(f"  [PASS] {name}: max |err| = {max_abs:.3e}, mean |err| = {mean_abs:.3e}")
        return True
    worst = int(np.argmax(np.abs(diff).reshape(-1)))
    particle, comp = divmod(worst, 3)
    print(f"  [FAIL] {name}: max |err| = {max_abs:.3e}, mean |err| = {mean_abs:.3e}")
    print(
        f"         worst at particle {particle}, component {comp}: "
        f"got {got.reshape(-1)[worst]:.6e}, expected {expected.reshape(-1)[worst]:.6e}"
    )
    return False


def _report_ele_mat(name, got, expected, index_cache):
    diff = got - expected
    block_err = np.abs(diff).reshape(diff.shape[0], -1).max(axis=1)
    max_abs = float(block_err.max())
    if np.allclose(got, expected, atol=ATOL, rtol=RTOL):
        print(f"  [PASS] {name}: max |err| = {max_abs:.3e}")
        return True
    worst = int(np.argmax(block_err))
    N = index_cache.N
    if worst < N:
        loc = f"diagonal block for particle {worst}"
    else:
        loc = f"off-diagonal block (index {worst})"
        for i in range(N):
            rp0, rp1 = index_cache.row_ptr[i]
            if rp0 <= worst < rp1:
                loc = f"off-diagonal block ({i}, {int(index_cache.col_indices[worst])})"
                break
    print(f"  [FAIL] {name}: max |err| = {max_abs:.3e} at {loc}")
    return False


def main():
    print(f"Loading scene from {SCENE_PATH}")
    model, _ = load_scene(str(SCENE_PATH))
    print(f"Loading ground truth from {GT_PATH}")
    gt = np.load(GT_PATH)

    index_cache = ParticleIndexCache(model)
    if _hash_index_map(index_cache.index_map) != int(gt["index_map_hash"]):
        print("ERROR: ParticleIndexCache layout does not match ground truth.")
        print("       The scene or builder has changed — ask your instructor for a fresh .npz.")
        sys.exit(2)

    results = []

    print("\nChecking eval_cloth_bending_forces ...")
    state = _state_from_gt(model, gt)
    try:
        eval_cloth_bending_forces(model, state)
    except NotImplementedError as e:
        print(f"  [SKIP] {e}")
        results.append(False)
    else:
        results.append(_report_array("bending forces", state.particle_f, gt["bending_f"]))

    print("\nChecking eval_cloth_stretch_shear_forces ...")
    state = _state_from_gt(model, gt)
    A = SparseParticleSystem(model, index_cache)
    try:
        eval_cloth_stretch_shear_forces(model, state, A, h=float(gt["h"]))
    except NotImplementedError as e:
        print(f"  [SKIP] {e}")
        results.extend([False, False])
    else:
        results.append(_report_array("stretch/shear forces", state.particle_f, gt["stretch_shear_f"]))
        results.append(
            _report_ele_mat(
                "stretch/shear Jacobian (A.ele_mat)",
                A.ele_mat,
                gt["stretch_shear_ele_mat"],
                index_cache,
            )
        )

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
