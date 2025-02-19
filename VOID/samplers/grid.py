import numpy as np

from .base import Sampler



class LocalizedGridSampler(Sampler):
    PARSER_NAME = "random"
    HELP = "Sample random points inside the unit cell of the given crystal structure"

    def __init__(self, lattice, centers = [], density=6, **kwargs):
        self.lattice = lattice
        self.centers = centers
        self.density = density

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--lattice",
            type=float,
            help="maximum number of points inside the crystal structure to sample (default: %(default)s)",
            default=5,
        )

    def get_points(self, structure):
        print(self.lattice)
        x = np.linspace(-self.lattice/2, self.lattice/2, self.density) # centered on zeo
        print(x)
        X, Y, Z = np.meshgrid(x, x, x)
        grid = np.vstack((X.flatten(), Y.flatten(),Z.flatten())).T
        for idx in self.centers:
            coor = structure.cart_coords[idx]
            if "fullgrid" not in locals():
                fullgrid = grid + coor
                print("coords: ", coor)
            else:
                fullgrid = np.vstack(fullgrid, grid + coor)
        elems = np.array([3 for _ in range(len(fullgrid))]).reshape(-1, 1)
        np.savetxt("points.xyz", np.hstack((elems, fullgrid)))
        return fullgrid