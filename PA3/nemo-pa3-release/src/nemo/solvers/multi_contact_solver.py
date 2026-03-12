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

        # Fill in the implementation here

        # TODO: Next, we enter into iterative solver
        # At this point, you should have predicted state (position and velocity) based on the instantaneous contacts.
        # and the preducted state is stored in state_out.
        for _ in range(self.maxits):
            # 1. Continuous collision detection
            # If there are no continuous contacts, we can exit the loop
            if len(contacts.contact_type_continuous) == 0:
                # print(f"BREAK t={t}")
                break
            # 2. Apply impulses and adjust the positions and velocities in state_out using collision impulses.
            # Hint: as described in the course note, you need to iterate through the continous contacts and
            # apply impulses sequentially (the order is not important). Before you apply each impulse, make sure
            # the contact is still valid (i.e., the contact is still approaching), as the previous collision impulse
            # in the loop may have changed the contact state.
            for i in range(len(contacts.contact_type_continuous)):

                # Replace the following line with your implementation.
                pass

        # NOTE: here the iteration goes up to self.maxits times. You are _not_ reuired to implement the geometric
        # collision response. But if you want to earn 10 (out of 100) bonus points, you are welcome to implement
        # the geometric collision response and the iteration to form impact zones, as discussed in the course note
        # and the lectures.
