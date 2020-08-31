# Import external packages

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys

#################
# Helper definitions
#################

# Parameters for the solvers
alpha = 1.4  # for the minmod limiter
eps = 0.000001  # when using sd2
odd = True  # when using fd2

# Index definitions for convenience
j0 = slice(2, -2)
jp = slice(3, -1)
jm = slice(1, -3)

# p coefficient
def p_coefs(u):
    pl0 = u[j0]
    pl1 = u[j0] - u[jm]
    pr0 = u[j0]
    pr1 = u[jp] - u[j0]
    pc0 = pl0 - (1.0 / 12.0) * (pr1 - pl1)
    pc1 = 0.5 * (pl1 + pr1)
    pc2 = pr1 - pl1
    return pl0, pl1, pr0, pr1, pc0, pc1, pc2


# Limiters (minmod, va)


def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def minmod3(a, b, c):
    return minmod(a, minmod(b, c))


def minmod_prime(u):
    return minmod3(
        alpha * (u[1:-1] - u[:-2]), 0.5 * (u[2:] - u[:-2]), alpha * (u[2:] - u[1:-1])
    )


def va(a, b):
    c = a ** 2 + b ** 2
    return np.divide(a * b * (a + b), c, out=np.zeros_like(a), where=c != 0)


def va_prime(u):
    return va(u[2:] - u[1:-1], u[1:-1] - u[:-2])


limiter = va_prime


# This is where the parameters are stored, along with the grid.
# One can solve different equations using the same setup.
# That's why this class definition is separate from Equation.
@dataclass
class Setup1d:
    # Grid parameters
    x_init: float = 0.0
    x_final: float = 1.0
    t_final: float = 1.0
    dt_out: float = 0.05
    J: int = 10
    cfl: float = 0.9
    dt: float = 0.0
    scheme: str = "sd3"  # can be fd2, sd2, or sd3


# This is the equation class including all equation-specific definitions.
class Equation1d(ABC):
    def __init__(self, setup):
        for key in setup.__dict__.keys():
            setattr(self, key, setup.__dict__[key])
        self.Nt = int(np.ceil(self.t_final / self.dt_out))
        self.x, self.dx = self.grid(self.x_init, self.x_final, self.J)

    def grid(self, x_init, x_final, J):
        dx = (x_final - x_init) / self.J
        x = np.linspace(x_init - 2.0 * dx, x_final + dx, J + 4)
        if self.scheme == "fd2":
            x += 0.5 * dx  # staggered grid for FD2
        return x, dx

    @abstractmethod
    def flux_x():
        pass

    @abstractmethod
    def initial_data():
        pass

    @abstractmethod
    def boundary_conditions():
        pass

    @abstractmethod
    def spectral_radius_x():
        pass


class Solver1d:
    def __init__(self, equation):
        for key in equation.__dict__.keys():
            setattr(self, key, equation.__dict__[key])

        # Equation dependent functions
        self.flux_x = equation.flux_x
        self.boundary_conditions = equation.boundary_conditions
        self.spectral_radius_x = equation.spectral_radius_x

        # # The unknown
        self.u = equation.initial_data()
        self.u_n = np.zeros((self.Nt + 1,) + self.u.shape)  # output array
        self.u_n[0] = self.u

    def H_flux(self, u_E, u_W, flux, spectral_radius):
        a = np.maximum(spectral_radius(u_E), spectral_radius(u_W))
        f_E = flux(u_E)
        f_W = flux(u_W)
        if u_W.shape == a.shape:
            return 0.5 * (f_W + f_E) - 0.5 * a * (u_W - u_E)  # scalar
        else:
            return 0.5 * (f_W + f_E) - 0.5 * np.multiply(
                a[:, None], (u_W - u_E)
            )  # for systems

    def c_flux(self, u_E, u_W):
        Hx_fluxp = self.H_flux(u_E[j0], u_W[jp], self.flux_x, self.spectral_radius_x)
        Hx_fluxm = self.H_flux(u_E[jm], u_W[j0], self.flux_x, self.spectral_radius_x)
        return -self.dt / self.dx * (Hx_fluxp - Hx_fluxm)

    #################
    # FD2
    #################

    def fd2(self, u):
        global odd
        u_prime = np.ones(u.shape)
        un_half = np.ones(u.shape)
        self.boundary_conditions(u)
        f = self.flux_x(u)
        u_prime[1:-1] = limiter(u)
        # Predictor
        un_half[1:-1] = u[1:-1] - 0.5 * self.dt / self.dx * limiter(f)
        f_half = self.flux_x(un_half)
        # Corrector
        if odd:
            u[1:-2] = (
                0.5 * (u[2:-1] + u[1:-2])
                + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
                - self.dt / self.dx * (f_half[2:-1] - f_half[1:-2])
            )
        else:
            u[2:-1] = (
                0.5 * (u[2:-1] + u[1:-2])
                + 0.125 * (u_prime[1:-2] - u_prime[2:-1])
                - self.dt / self.dx * (f_half[2:-1] - f_half[1:-2])
            )
        # Boundary conditions
        self.boundary_conditions(u)
        # Switch
        odd = not odd
        return u

    #################
    # SD2
    #################

    def reconstruction_sd2(self, u):
        # Reconstruction
        u_E = np.ones(u.shape)
        u_W = np.ones(u.shape)
        s = limiter(u[1:-1])
        u_E[j0] = u[j0] + 0.5 * s
        u_W[j0] = u[j0] - 0.5 * s
        self.boundary_conditions(u_E)
        self.boundary_conditions(u_W)
        return u_E, u_W

    def sd2(self, u):
        self.boundary_conditions(u)
        u_E, u_W = self.reconstruction_sd2(u)
        C0 = self.c_flux(u_E, u_W)
        u[j0] += C0
        self.boundary_conditions(u)
        u_E, u_W = self.reconstruction_sd2(u)
        C1 = self.c_flux(u_E, u_W)
        u[j0] += 0.5 * (C1 - C0)
        self.boundary_conditions(u)
        return u

    #################
    # SD3
    #################

    def reconstruction_sd3(self, u, ISl, ISc, ISr):
        cl = 0.25
        cc = 0.5
        cr = 0.25
        alpl = cl / ((eps + ISl) * (eps + ISl))
        alpc = cc / ((eps + ISc) * (eps + ISc))
        alpr = cr / ((eps + ISr) * (eps + ISr))
        alp_sum = alpl + alpc + alpr
        wl = alpl / alp_sum
        wc = alpc / alp_sum
        wr = alpr / alp_sum
        pl0, pl1, pr0, pr1, pc0, pc1, pc2 = p_coefs(u)
        u_E = np.ones(u.shape)
        u_W = np.ones(u.shape)
        u_E[j0] = (
            wl * (pl0 + 0.5 * pl1)
            + wc * (pc0 + 0.5 * pc1 + 0.25 * pc2)
            + wr * (pr0 + 0.5 * pr1)
        )
        u_W[j0] = (
            wl * (pl0 - 0.5 * pl1)
            + wc * (pc0 - 0.5 * pc1 + 0.25 * pc2)
            + wr * (pr0 - 0.5 * pr1)
        )
        # boundary
        self.boundary_conditions(u_E)
        self.boundary_conditions(u_W)
        return u_E, u_W

    # Implemented for one equation (for systems modify IS declarations)
    def sd3(self, u):
        self.boundary_conditions(u)
        u_norm = np.sqrt(self.dx) * np.linalg.norm(u[j0])
        pl0, pl1, pr0, pr1, pc0, pc1, pc2 = p_coefs(u)
        ISl = pl1 * pl1 / (u_norm + eps)
        ISc = 1.0 / (u_norm + eps) * ((13.0 / 3.0) * pc2 * pc2 + pc1 * pc1)
        ISr = pr1 * pr1 / (u_norm + eps)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr)
        C0 = self.c_flux(u_E, u_W)
        u[2:-2] += +C0
        self.boundary_conditions(u)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr)
        C1 = self.c_flux(u_E, u_W)
        u[j0] += +0.25 * (C1 - 3.0 * C0)
        self.boundary_conditions(u)
        u_E, u_W = self.reconstruction_sd3(u, ISl, ISc, ISr)
        C2 = self.c_flux(u_E, u_W)
        u[j0] += +1.0 / 12.0 * (8.0 * C2 - C1 - C0)
        self.boundary_conditions(u)
        return u

    # Main solver routine
    def solve(self):
        if self.scheme == "sd2":
            self.step = self.sd2
        elif self.scheme == "sd3":
            self.step = self.sd3
        elif self.scheme == "fd2":
            self.step = self.fd2
        else:
            sys.exit(
                "Scheme "
                + self.scheme
                + " is not recognized! Choices are: fd2, sd2, sd3."
            )
        i = 0
        t = 0.0
        t_out = 0.0
        while t < self.t_final:
            r_max = np.max(self.spectral_radius_x(self.u))
            dt = self.dx * self.cfl / r_max
            self.dt = min(dt, self.dt_out - t_out)
            t += self.dt
            t_out += self.dt
            self.u = self.step(self.u)
            # Store if t_out=dt_out
            if t_out == self.dt_out:
                i += 1
                self.u_n[i, :] = self.u
                t_out = 0
