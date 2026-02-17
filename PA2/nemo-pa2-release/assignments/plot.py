class PlotSpec:
    def __init__(self, config: dict):
        self.particle_id = config["particle_id"]
        self.dof = config["dof"]
        v = config["y_range"]
        self.y_range_min = v[0]
        self.y_range_max = v[1]
