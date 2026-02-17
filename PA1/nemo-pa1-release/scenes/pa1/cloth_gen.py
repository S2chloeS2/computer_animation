import yaml

# Generate a MxN cloth
N = 10
M = 20


def main():
    data = {"solver": {"type": "symplectic_euler", "timestep": 0.0005}}

    particles = []
    springs = []
    dx = 0.005
    cnt = 0
    for i in range(N):
        for j in range(M):
            p = {
                "mass": 0.1,
                "vel": (0, 0, 0),
                "radius": 0.01,
                "pos": (i * dx, j * dx, 4),
            }
            if j == 0 or j == M-1:
                p["fixed"] = True
            particles.append(p)
            cnt += 1

            if j > 0:
                s = {
                    "particle_ids": (cnt-2, cnt-1),
                    "stiffness": 3500.0,
                    "damping": 1.8,
                }
                springs.append(s)
            if i > 0:
                s = {
                    "particle_ids": (cnt-1-M, cnt-1),
                    "stiffness": 300.0,
                    "damping": 0.1,
                }
                springs.append(s)

    data["particles"] = particles
    data["springs"] = springs

    with open("out.yml", "w") as file:
        yaml.safe_dump(data, file, sort_keys=False)


if __name__ == "__main__":
    main()
