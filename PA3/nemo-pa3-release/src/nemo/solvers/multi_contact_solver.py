import numpy as np
from rich import print as rprint

from ..core.types import override
from ..geometry.collision import CollisionDetector
from ..geometry.types import ParticleFlags
from ..sim.contacts import ContactType
from ..sim.forces import eval_all_forces
from ..sim.model import Model
from ..sim.state import State
from .contact_penalty_solver import penalty_force
from .integrator import IntegratorBase
from .solver import SolverBase
from .symplectic_euler import SymplecticEulerSolver


class MultiContactSolver(SolverBase):
    """Contact solver that considers multiple contacts.

    This solver responds to contacts using an iterative approach that we discussed in class.
    """

    def __init__(
        self,
        model: Model,
        dt: float,
        maxits: int | None = None,
        stiffness: float | None = None,
        damping: float | None = None,
        integrator: IntegratorBase | None = None,
    ):
        """
        Initialize the iterative contact solver.

        Args:
            model: The model to simulate.
            dt: The time step.
            maxits: The maximum number of iterations to resolve contact before resorting to the
                    gemetric collision response.
            stiffness: The stiffness coefficient for the penalty force.
            damping: The damping coefficient for the penalty force.
            integrator: The time integrator to use.

        NOTE: The stiffness and damping coefficients are used to control the penalty force
              between particles. THe penalty force between particle and fixed shapes is controlled
              by `shape_penalty_params` in the model.
        """
        super().__init__(model=model, dt=dt)

        if integrator is None:
            integrator = SymplecticEulerSolver(model, dt)
        elif integrator.model is not model:
            # check model consistency
            raise RuntimeError("model in the provided solver and the given model must be the same.")
        self.integrator = integrator

        if maxits is None:
            maxits = 8
        elif maxits < 0:
            raise ValueError("maxits must be non-negative")

        self.maxits = maxits
        self.stiffness = stiffness if stiffness is not None else 1000.0
        self.damping = damping if damping is not None else 0.0
        rprint("  Use [bold green]Multi-Contact Solver[/bold green]")
        rprint(f"  Particle Contact Penalty: Stiffness={self.stiffness}, Damping={self.damping}")
        rprint(f"  Multi-Contact Solver: maxits={self.maxits}")
        self.collision = CollisionDetector(model)

        mask = self.model.particle_flags & ParticleFlags.ACTIVE.value != 0
        self.masked_mass = np.where(mask, self.model.particle_mass, 0.0)

    @override
    def step(self, state_in: State, state_out: State, dt: float | None = None):
        """Step the contact with an iterative solver.

        Args:
            state_in: The input state.
            state_out: The output state.
            contacts: The contact information.
            dt: The time step.

        NOTE: the contacts must be updated before calling this method
              by the collision detection algorithm (see geometry/collision.py).
              The contacts must be updated based on state_in (not state_out).
        """
        if dt is None:
            dt = self.dt
        self.ts += dt

        # Detect collisions
        contacts = self.collision.instantaneous_contacts(state_in)
        # TODO: First, we integrate the system by considering instantaneous contacts and penalty forces.
        # This is exactly the same as the ContactPenaltySolver.

        # Evaluate the penalty forces. The contacts are not updated in this function, and thus
        # the penalty force is evaluated only once here.
        f_penalty = np.zeros_like(state_in.particle_f)

        for i in range(len(contacts.contact_type)):
            if contacts.contact_type[i] == ContactType.PARTICLE_PARTICLE:
                par0_id = contacts.contact_instance0[i]
                par1_id = contacts.contact_instance1[i]
                n = contacts.contact_normal[i]
                depth = np.dot(contacts.contact_point1[i] - contacts.contact_point0[i], n)
                if depth < 0.0:
                    v_rel = np.dot(state_in.particle_qd[par1_id] - state_in.particle_qd[par0_id], n)
                    F = penalty_force(depth, v_rel, self.stiffness, self.damping)
                    if self.model.particle_flags[par1_id] & ParticleFlags.ACTIVE.value != 0:
                        f_penalty[par1_id] += F * n
                    if self.model.particle_flags[par0_id] & ParticleFlags.ACTIVE.value != 0:
                        f_penalty[par0_id] -= F * n
            elif contacts.contact_type[i] == ContactType.FIXED_SHAPE_PARTICLE:
                shape_id = contacts.contact_instance0[i]
                par_id = contacts.contact_instance1[i]
                depth = np.dot(contacts.contact_point1[i] - contacts.contact_point0[i], contacts.contact_normal[i])
                if depth < 0.0:
                    n = contacts.contact_normal[i]
                    stiffness = self.model.shape_penalty_params[shape_id, 0]
                    damping = self.model.shape_penalty_params[shape_id, 1]
                    v_rel = np.dot(state_in.particle_qd[par_id], n)
                    F = penalty_force(depth, v_rel, stiffness, damping)
                    f_penalty[par_id] += F * n

        def f(model: Model, state: State) -> None:
            eval_all_forces(model, state)
            state.particle_f += np.outer(self.masked_mass, self.model.gravity)
            state.particle_f += f_penalty

        self.integrator.integrate(state_in, state_out, f, dt)

        # Iterative CCD loop
        for _ in range(self.maxits):
            # 1. Continuous collision detection
            self.collision.continuous_contacts(state_in, state_out, contacts)
            # If there are no continuous contacts, we can exit the loop
            if len(contacts.contact_type_continuous) == 0:
                # print(f"BREAK t={t}")
                break
            # 2. Apply impulses and adjust the positions and velocities in state_out using collision impulses.
            for i in range(len(contacts.contact_type_continuous)):
                ctype = contacts.contact_type_continuous[i]
                n = contacts.contact_normal_continuous[i]
                if ctype == ContactType.FIXED_SHAPE_PARTICLE:
                    shape_id = contacts.contact_instance0_continuous[i]
                    par_id = contacts.contact_instance1_continuous[i]
                    v_n = np.dot(state_out.particle_qd[par_id], n)
                    if v_n >= 0.0:
                        continue  # not approaching, skip
                    e = min(
                        self.model.particle_restitution_coeff[par_id],
                        self.model.shape_restitution_coeff[shape_id],
                    )
                    dv = -(1.0 + e) * v_n
                    state_out.particle_qd[par_id] += dv * n
                    state_out.particle_q[par_id] += dv * n * dt
                elif ctype == ContactType.PARTICLE_PARTICLE:
                    par0_id = contacts.contact_instance0_continuous[i]
                    par1_id = contacts.contact_instance1_continuous[i]
                    v_rel_n = np.dot(state_out.particle_qd[par1_id] - state_out.particle_qd[par0_id], n)
                    if v_rel_n >= 0.0:
                        continue  # not approaching, skip
                    e = min(
                        self.model.particle_restitution_coeff[par0_id],
                        self.model.particle_restitution_coeff[par1_id],
                    )
                    inv_mass0 = self.model.particle_inv_mass[par0_id]
                    inv_mass1 = self.model.particle_inv_mass[par1_id]
                    total_inv_mass = inv_mass0 + inv_mass1
                    if total_inv_mass < 1e-12:
                        continue
                    J = -(1.0 + e) * v_rel_n / total_inv_mass
                    if self.model.particle_flags[par0_id] & ParticleFlags.ACTIVE.value != 0:
                        state_out.particle_qd[par0_id] -= J * inv_mass0 * n
                        state_out.particle_q[par0_id] -= J * inv_mass0 * n * dt
                    if self.model.particle_flags[par1_id] & ParticleFlags.ACTIVE.value != 0:
                        state_out.particle_qd[par1_id] += J * inv_mass1 * n
                        state_out.particle_q[par1_id] += J * inv_mass1 * n * dt

        # NOTE: here the iteration goes up to self.maxits times. You are _not_ reuired to implement the geometric
        # collision response. But if you want to earn 10 (out of 100) bonus points, you are welcome to implement
        # the geometric collision response and the iteration to form impact zones, as discussed in the course note
        # and the lectures.
