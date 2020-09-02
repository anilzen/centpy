# This is the equation class including all equation-specific definitions.

import numpy as np
from abc import ABC, abstractmethod
from .parameters import Pars1d, Pars2d

# 1d
class Equation1d(ABC, Pars1d):
    def __init__(self, pars):
        for key in pars.__dict__.keys():
            setattr(self, key, pars.__dict__[key])
        self.Nt = int(np.ceil(self.t_final / self.dt_out))
        self.x, self.dx = self.grid(self.x_init, self.x_final, self.J)

        # Boolean for FD2
        self.odd = True

    def grid(self, x_init, x_final, J):
        dx = (x_final - x_init) / self.J
        x = np.linspace(x_init - 2.0 * dx, x_final + dx, J + 4)
        if self.scheme == "fd2":
            x += 0.5 * dx  # staggered grid for FD2
        return x, dx

    @abstractmethod
    def initial_data(self):
        pass

    @abstractmethod
    def boundary_conditions(self, u):
        pass

    @abstractmethod
    def flux_x(self, u):
        pass

    @abstractmethod
    def spectral_radius_x(self, u):
        pass


# 2d
class Equation2d(Equation1d, Pars2d):
    def __init__(self, pars):
        super().__init__(pars)
        self.y, self.dy = self.grid(self.y_init, self.y_final, self.K)
        self.xx, self.yy = np.meshgrid(self.x, self.y, sparse=True)

    @abstractmethod
    def flux_y(self, u):
        pass

    @abstractmethod
    def spectral_radius_y(self, u):
        pass

