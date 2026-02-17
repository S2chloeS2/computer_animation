import numpy as np

from nemo.sim import ModelBuilder
from nemo.sim.forces import (
    eval_all_force_pos_jacobians,
    eval_all_forces,
    eval_drag_force_vel_jacobians,
    eval_drag_forces,
    eval_gravitational_force_pos_jacobians,
    eval_gravitational_forces,
    eval_spring_force_pos_jacobians,
    eval_spring_force_vel_jacobians,
    eval_spring_forces,
)


def test_drag_force_vel_jacobians():
    builder = ModelBuilder()
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0, drag=0.4)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_drag_force_vel_jacobians(model, state, A, S)

    EPS = 1e-6
    state.clear_forces()
    state.particle_qd[0, 0] = EPS
    eval_drag_forces(model, state)
    f_p = state.particle_f.copy()

    state.clear_forces()
    state.particle_qd[0, 0] = -EPS
    eval_drag_forces(model, state)
    f_m = state.particle_f.copy()
    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac * S, A[:, 0])

    state.particle_qd[0, 0] = 0

    state.clear_forces()
    state.particle_qd[0, 1] = EPS
    eval_drag_forces(model, state)
    f_p = state.particle_f.copy()

    state.clear_forces()
    state.particle_qd[0, 1] = -EPS
    eval_drag_forces(model, state)
    f_m = state.particle_f.copy()
    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac * S, A[:, 1])

    state.particle_qd[0, 1] = 0

    state.clear_forces()
    state.particle_qd[0, 2] = EPS
    eval_drag_forces(model, state)
    f_p = state.particle_f.copy()

    state.clear_forces()
    state.particle_qd[0, 2] = -EPS
    eval_drag_forces(model, state)
    f_m = state.particle_f.copy()
    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac * S, A[:, 2])


def test_gravitational_force_pos_jacobians():
    builder = ModelBuilder()
    builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_particle(pos=(1, 0.2, 1.2), vel=(0, 0, 0), mass=1.2)
    builder.add_gravitational(0, 1, 1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_gravitational_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    state.clear_forces()
    t = state.particle_q[0, 0]
    state.particle_q[0, 0] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[0, 0] = t
    state.clear_forces()
    state.particle_q[0, 0] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[0, 0] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 0])

    # Another column
    state.clear_forces()
    t = state.particle_q[1, 1]
    state.particle_q[1, 1] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 1] = t
    state.clear_forces()
    state.particle_q[1, 1] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_q[1, 2]
    state.particle_q[1, 2] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 2] = t
    state.clear_forces()
    state.particle_q[1, 2] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 5])


def test_gravitational_force_pos_jacobians2():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(0, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(1, 0.2, 1.2), vel=(0, 0, 0), mass=1.2)
    builder.add_gravitational(0, 1, 1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_gravitational_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_q[1, 1]
    state.particle_q[1, 1] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 1] = t
    state.clear_forces()
    state.particle_q[1, 1] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_q[1, 2]
    state.particle_q[1, 2] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 2] = t
    state.clear_forces()
    state.particle_q[1, 2] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 5])


def test_gravitational_force_pos_jacobians3():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.0, 0.0, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(0, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(1, 0.2, 1.2), vel=(0, 0, 0), mass=1.2)
    builder.add_gravitational(0, 1, 1.0)
    builder.add_gravitational(1, 2, 2.0)
    builder.add_gravitational(0, 2, 1.5)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_gravitational_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_q[2, 1]
    state.particle_q[2, 1] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[2, 1] = t
    state.clear_forces()
    state.particle_q[2, 1] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[2, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 7])

    # yet another column
    state.clear_forces()
    t = state.particle_q[0, 2]
    state.particle_q[0, 2] += EPS
    eval_gravitational_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[0, 2] = t
    state.clear_forces()
    state.particle_q[0, 2] -= EPS
    eval_gravitational_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[0, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 2])


def test_spring_force_pos_jacobians():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(1, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(1, 0.2, 1.2), vel=(2, 0, 0), mass=1.2)
    builder.add_spring(0, 1, ke=1.2, kd=0.0, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_spring_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_q[1, 1]
    state.particle_q[1, 1] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 1] = t
    state.clear_forces()
    state.particle_q[1, 1] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_q[1, 2]
    state.particle_q[1, 2] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 2] = t
    state.clear_forces()
    state.particle_q[1, 2] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 5])


def test_spring_force_pos_jacobians2():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(1, 0, 0), mass=1.0, flags=0)
    builder.add_particle(pos=(1, 0.2, 1.2), vel=(2, 0, 0), mass=1.2)
    builder.add_spring(0, 1, ke=0.0, kd=1.2, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_spring_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_q[1, 1]
    state.particle_q[1, 1] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 1] = t
    state.clear_forces()
    state.particle_q[1, 1] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_q[1, 2]
    state.particle_q[1, 2] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 2] = t
    state.clear_forces()
    state.particle_q[1, 2] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 5])


def test_spring_force_pos_jacobians3():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(1, 0, 0), mass=1.0)
    builder.add_particle(pos=(1.3, -0.2, 1.2), vel=(2, 0, 0), mass=1.2)
    builder.add_spring(0, 1, ke=1.0, kd=1.2, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_spring_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_q[1, 1]
    state.particle_q[1, 1] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[1, 1] = t
    state.clear_forces()
    state.particle_q[1, 1] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_q[0, 2]
    state.particle_q[0, 2] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[0, 2] = t
    state.clear_forces()
    state.particle_q[0, 2] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[0, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 2])


def test_spring_force_vel_jacobians():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(1, 0, 0), mass=1.0)
    builder.add_particle(pos=(1.3, -0.2, 1.2), vel=(2, 0, 0), mass=1.2)
    builder.add_spring(0, 1, ke=1.0, kd=1.2, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_spring_force_vel_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # Another column
    state.clear_forces()
    t = state.particle_qd[1, 1]
    state.particle_qd[1, 1] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_qd[1, 1] = t
    state.clear_forces()
    state.particle_qd[1, 1] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_qd[1, 1] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 4])

    # yet another column
    state.clear_forces()
    t = state.particle_qd[0, 2]
    state.particle_qd[0, 2] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_qd[0, 2] = t
    state.clear_forces()
    state.particle_qd[0, 2] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_qd[0, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 2])


def test_spring_force_vel_jacobians2():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(0, 0, 0), mass=1.0)
    builder.add_particle(pos=(1.3, -0.2, 1.2), vel=(0, 0, 0), mass=1.2, flags=0)
    builder.add_spring(0, 1, ke=1.0, kd=1.2, rest_length=1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.3
    eval_spring_force_vel_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # yet another column
    state.clear_forces()
    t = state.particle_qd[0, 2]
    state.particle_qd[0, 2] += EPS
    eval_spring_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_qd[0, 2] = t
    state.clear_forces()
    state.particle_qd[0, 2] -= EPS
    eval_spring_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_qd[0, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 2])


def test_all_force_pos_jacobians():
    builder = ModelBuilder()
    builder.add_particle(pos=(0.1, 0.2, 0), vel=(0, 0, 0), mass=1.0, drag=0.4)
    builder.add_particle(pos=(1.3, -0.2, 1.2), vel=(0, 0, 0), mass=1.2, flags=0)
    builder.add_spring(0, 1, ke=1.0, kd=1.2, rest_length=1.0)
    builder.add_gravitational(0, 1, 1.0)
    model = builder.finalize()
    state = model.state()
    A = np.zeros((model.particle_count * 3, model.particle_count * 3))
    S = 0.5
    eval_all_force_pos_jacobians(model, state, A, S)

    # finite difference approximation of the jacobian
    EPS = 1e-6
    # yet another column
    state.clear_forces()
    t = state.particle_q[0, 2]
    state.particle_q[0, 2] += EPS
    eval_all_forces(model, state)
    f_p = state.particle_f.copy()

    state.particle_q[0, 2] = t
    state.clear_forces()
    state.particle_q[0, 2] -= EPS
    eval_all_forces(model, state)
    f_m = state.particle_f.copy()
    state.particle_q[0, 2] = t

    jac = (f_p - f_m) / (2 * EPS)  # finite difference approximation of the jacobian
    assert np.allclose(jac.reshape(-1) * S, A[:, 2])
