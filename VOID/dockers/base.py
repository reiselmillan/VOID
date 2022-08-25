import numpy as np
from typing import List
from pymatgen.core import Structure, Molecule

from VOID.structure import Complex
from VOID.object import ParseableObject


ATTEMPTS = 50


class Docker(ParseableObject):
    """Base class to dock a guest into a crystal"""

    PARSER_NAME = "base"
    HELP = "Base docker; does not implement any docking procedure"

    def __init__(self, host, guest, sampler, fitness, **kwargs):
        self.host = host
        self.guest = guest
        self.sampler = sampler
        self.fitness = fitness

        self.constraint = kwargs.get("constraint", [])
        self.sphere = kwargs.get("sphere", 5)
        self.max_num_of_sites = kwargs.get("max_num_sites", None)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--attempts",
            type=int,
            help="maximum number of attempts to dock (default: %(default)s)",
            default=ATTEMPTS,
        )

    def copy(self):
        return self.__class__(
            self.host.copy(), self.guest.copy(), self.sampler, self.fitness
        )

    def new_host(self, newcoords=None):
        if newcoords is None:
            return self.host.copy()

        return Structure(
            species=self.host.species,
            coords=newcoords,
            lattice=self.host.lattice.matrix,
            coords_are_cartesian=True,
        )

    def new_guest(self, newcoords=None):
        if newcoords is None:
            return self.guest.copy()

        return Molecule(species=self.guest.species, coords=newcoords,)

    def create_new_complex(self, host_coords, guest_coords):
        return Complex(
            self.new_host(newcoords=host_coords),
            self.new_guest(newcoords=guest_coords),
            add_transform=False
        )

    def apply_constraint(self, points):
        if not self.constraint:
            return points

        diff = points - self.host.cart_coords[self.constraint]
        diff2 = (diff * diff).sum(axis=1)
   
        filtered_inds = np.where(diff2 < self.sphere**2)
        if self.max_num_of_sites is not None and self.max_num_of_sites < len(filtered_inds):
            return points[filtered_inds][:self.max_num_of_sites]
        return points[filtered_inds]

    def dock(self, attempts: int) -> List[Complex]:
        """Docks the guest into the host.
        """
        complexes = []
        # apply constraint
        points = self.sampler.get_points(self.host)
        points = self.apply_constraint(points)
        for point in points:
            complexes += self.dock_at_point(point, attempts)

        complexes = self.rank_complexes(complexes)

        return complexes

    def dock_at_point(self, point, attempts):
        raise NotImplementedError

    def rank_complexes(self, complexes):
        scores = [self.fitness(cpx) for cpx in complexes]
        ranking = sorted(zip(complexes, scores), key=lambda x: x[1], reverse=True)

        return [cpx for cpx, fit in ranking if fit >= 0]
