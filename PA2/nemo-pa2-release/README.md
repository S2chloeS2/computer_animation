# Physics-based Simulator for Education (COMS W4167)
## Setup your Python deveopment environment
The projects are based on Python. So you need to install a few python packages. 
While you can use the Python's native `pip` tool to include packages, we highly 
recommdn you to use `uv`, which is much faster than `pip` and has been widely used in industry.
1. Install `uv`on your computer following the [instruction](https://docs.astral.sh/uv/getting-started/installation/).

2. After that, you need to create a Python virtual environment. We suggest you create a Python virtual enviornment separted from other Python environments on your computer (this way, it's clearn and it won't mess up your system's Python installation). For that, open a terminal and go to `nemo` directory, and run
```
uv venv --python 3.13
source .venv/bin/activate
```
You can also use Python 3.12 (while here I just choose Python 3.13).
The second line above is to activate your newly installed Python environment (located in the local directory `.venv` within `nemo` directory).

3.  To install dependencies, simply run
```
uv sync -U
```
And you are ready to go!


## Run simulation for individual assignments
- PA1:
```
python -m assignments.run pa1 scenes/pa1/scene01.yml
```
We provided a set of scene files in `scenes/pa1` for students to test their implementations. The scene files are in [`.ymal` format](https://yaml.org/),
specifying the time integrator solver, timestep size, and the initial state of 
the simulated scene.
Please refer to the comments in the scene file (e.g., `scenes/pa1/scene00.yml`)
to what needs to be configured to launch a simulation.

## Code Overview
The code structure is similar to [Nvidia Newton](https://github.com/newton-physics/newton). A high-level philosophy we follow is to sperate simulated **scene** from the 
simulation **state**. A simulated **scene** is stored in `sim.model.Model`, describing how many objects are in the scene, their starting positions, spring stiffnesses, and other information (see `src/nemo/sim/model.py`)---this information stay unchanged throughout the entire simulation. Simulation **state**, in contrast, includes data that will change over time---for example, particle positions, velocities, and forces (see `src/nemo/sim/state.py`).

Another high-level idea we take in this codebase is that we don't have Python classes
that correspond to indivdual simulated objects. For example, we don't have a _Particle_ class that maintains particle position, velocity, and force. Instead, we group all particle positions into a Nx3 array ([numpy array](https://numpy.org/)), where N is the number of particles. Similarly, all particles' velocities are stored in another Nx3 array. These arrays are all maintained in `sim.state.State`, and will be updated by a _solver_ at each time step. This data structure is to facilitate paralell processing (i.e., more GPU friendly). For example, you can process all rows of the Nx3 array of position/velocity in parallel (GPU or CPU) threads. 

NOTE: For this course assignments, we don't ask students to implement parallel algorithms. To finish assignments, it is ok to `for` loop through all rows of state arrays (meaning processing each simulated objects sequentially).

The major steps of launch a simulation is as follows:
1. Create a `ModelBuilder`, and use the bulider to populate a simulated scene, such as adding individual particles, connecting particles with springs and adding rigid bodies.
2. Create a solver (e.g., a `ExplicitEulerSolver`)
3. Create a contact detector and other external force generateor (not needed until the 3rd PA)
4. In each timestep, detect collisions and advance the state using the solver.

For example, for PA1, a model and a solver are created in `assignments.pa1.load_scene`, and the timestepping happens in the while loop in `assignments.run.Runner.launch` method.


## PA2 Bonus: Creative Scenes

Two custom scenes are provided under `scenes/pa2/`. Both use only the existing
spring forces already implemented in PA2 (no code changes required).

---

### Scene 1: Rope Bridge (`scenes/pa2/scene07_rope_bridge.yml`)

**Run command:**
```
python3 -m assignments.run pa2 scenes/pa2/scene07_rope_bridge.yml
```

**Physical setup:**
9 particles are placed horizontally at equal spacing (0.5 m) along the x-axis
at height z=2.0. The leftmost particle (ID 0) and rightmost particle (ID 8)
are fixed as anchors. The 7 free particles in between are connected by springs
with rest length 0.5 m (matching the initial spacing) and moderate stiffness
(k=200). Under the default gravity field (−9.81 m/s² in z), the free particles
sag downward and the chain settles into a **catenary (hanging chain) shape** —
the classic equilibrium of a flexible rope under gravity.

**Why it is interesting for implicit integration:**
With k=200 and h=0.005, the stiffness-to-timestep ratio (k·h²=0.005) is already
in a regime where explicit Euler would exhibit growing oscillations. The
`LinearizedImplicitSolver` handles this stably in a single linear solve per
step, allowing the bridge to smoothly settle to its catenary equilibrium. The
real-time plot shows the midpoint particle (ID 4) dropping from z=2.0 and
converging to its equilibrium height with nicely damped oscillations.

---

### Scene 2: Pendulum Chain (`scenes/pa2/scene08_pendulum_chain.yml`)

**Run command:**
```
python3 -m assignments.run pa2 scenes/pa2/scene08_pendulum_chain.yml
```

**Physical setup:**
6 particles are arranged vertically with 0.5 m spacing. Particle 0 (top, z=3.0)
is fixed as the pivot. Particles 1–5 hang below in a chain, each connected to
the one above by a spring (k=80, rest length 0.5 m, damping β=0.15). All free
particles are given an initial horizontal velocity of vx=0.3 m/s, which starts
the chain swinging like a pendulum. Because particles further from the pivot
lag behind, the chain traces a characteristic **whip shape** during the first
swing before gradually settling into synchronized oscillation.

**Why it is interesting for implicit integration:**
A 5-link pendulum chain is a highly nonlinear, coupled system. The gravitational
restoring force and spring tension interact across all links simultaneously.
With h=0.005, explicit Euler is unstable for this setup (the chain rapidly
diverges). The `ImplicitEulerSolver` (full Newton iteration) resolves the
nonlinear coupling accurately at each step, maintaining stable damped oscillation
throughout the simulation. The plot of the chain tip (particle 5) z-position
clearly shows the decaying pendulum amplitude — a direct demonstration that
the implicit integrator correctly captures energy dissipation in a multi-body
coupled spring system.