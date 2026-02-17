import numpy as np
import polyscope as ps

# Initialize Polyscope
ps.init()

# Define the nodes of the curve network (e.g., the two endpoints of the cylinder's central axis)
nodes = np.array(
    [
        [0.0, 0.0, 0.0],  # Bottom center
        [0.0, 0.0, 5.0],  # Top center
    ]
)

# Define the edges connecting the nodes
edges = np.array([[0, 1]])

vertices = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],  # Bottom face
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],  # Top face
    ],
    dtype=float,
)
vertices[:, 1] += 1

faces = np.array(
    [
        [0, 3, 2, 1],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [0, 4, 7, 3],  # Left face
    ],
    dtype=int,
)

# Register the curve network
ps_net = ps.register_curve_network("my_cylinder_axis", nodes, edges)

# Set the radius of the cylinder (optional, default is usually fine)
ps_net.set_radius(0.1)

ps_mesh = ps.register_surface_mesh("My Cube", vertices, faces)

# Show the Polyscope window
ps.show()
