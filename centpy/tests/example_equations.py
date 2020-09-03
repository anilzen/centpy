# Class definitions for equations to be tested

from path import *
import numpy as np
import centpy

###############
# 1D equations
###############

# Burgers equation
class Burgers1d(centpy.Equation1d):
    def initial_data(self):
        return np.sin(self.x) + 0.5 * np.sin(0.5 * self.x)

    def boundary_conditions(self, u):
        u[0] = u[-4]
        u[1] = u[-3]
        u[-2] = u[2]
        u[-1] = u[3]

    def flux_x(self, u):
        return 0.5 * u * u

    def spectral_radius_x(self, u):
        return np.abs(u)


# Euler equation
class Euler1d(centpy.Equation1d):
    def initial_data(self):
        u = np.zeros((self.J + 4, 3))
        midpoint = int(self.J / 2) + 2

        left_v = [1, 0, 1.0 / (self.gamma - 1.0)]
        right_v = [0.125, 0.0, 0.1 / (self.gamma - 1.0)]

        # Left side
        u[:midpoint, :] = left_v
        # Right side
        u[midpoint:, :] = right_v

        return u

    def boundary_conditions(self, u):
        left_v = [1, 0, 1.0 / (self.gamma - 1.0)]
        right_v = [0.125, 0.0, 0.1 / (self.gamma - 1.0)]
        # Left side
        u[0] = left_v
        u[1] = left_v
        # Right side
        u[-1] = right_v
        u[-2] = right_v

    def flux_x(self, u):
        f = np.zeros_like(u)
        rho = u[:, 0]
        u_x = u[:, 1] / rho
        E = u[:, 2]
        p = (self.gamma - 1.0) * (E - 0.5 * rho * u_x ** 2)

        f[:, 0] = rho * u_x
        f[:, 1] = rho * u_x ** 2 + p
        f[:, 2] = u_x * (E + p)

        return f

    def spectral_radius_x(self, u):
        rho = u[:, 0]
        u_x = u[:, 1] / rho
        p = (self.gamma - 1.0) * (u[:, 2] - 0.5 * rho * u_x ** 2)
        return np.abs(u_x) + np.sqrt(self.gamma * p / rho)


# MHD equation
class MHD1d(centpy.Equation1d):
    def pressure(self, u):
        return (
            u[:, 6]
            - 0.5 * ((u[:, 1] ** 2 + u[:, 2] ** 2 + u[:, 3] ** 2) / u[:, 0])
            - 0.5 * (self.B1 ** 2 + u[:, 4] ** 2 + u[:, 5] ** 2)
        )

    def initial_data(self):
        u = np.zeros((self.J + 4, 7))
        midpoint = int(self.J / 2) + 2

        # Left side
        u[:midpoint, 0] = 1.0
        u[:midpoint, 1] = 0.0
        u[:midpoint, 2] = 0.0
        u[:midpoint, 3] = 0.0
        u[:midpoint, 4] = 1.0
        u[:midpoint, 5] = 0.0
        u[:midpoint, 6] = 1.0 + 25.0 / 32.0

        # Right side
        u[midpoint:, 0] = 0.125
        u[midpoint:, 1] = 0.0
        u[midpoint:, 2] = 0.0
        u[midpoint:, 3] = 0.0
        u[midpoint:, 4] = -1.0
        u[midpoint:, 5] = 0.0
        u[midpoint:, 6] = 0.1 + 25.0 / 32.0

        return u

    def boundary_conditions(self, u):

        left_v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0 + 25.0 / 32.0]
        right_v = [0.125, 0.0, 0.0, 0.0, -1.0, 0.0, 0.1 + 25.0 / 32]

        if self.odd:
            u[0] = left_v
            u[-1] = right_v
            u[-2] = right_v
        else:
            u[0] = left_v
            u[1] = left_v
            u[-1] = right_v

    def flux_x(self, u):
        f = np.zeros_like(u)

        B1 = self.B1

        p_star = self.pressure(u) + 0.5 * (B1 ** 2 + u[:, 4] ** 2 + u[:, 5] ** 2)

        f[:, 0] = u[:, 1]
        f[:, 1] = u[:, 1] ** 2 / u[:, 0] + p_star
        f[:, 2] = u[:, 1] * u[:, 2] / u[:, 0] - B1 * u[:, 4]
        f[:, 3] = u[:, 1] * u[:, 3] / u[:, 0] - B1 * u[:, 5]
        f[:, 4] = u[:, 1] * u[:, 4] / u[:, 0] - B1 * u[:, 2] / u[:, 0]
        f[:, 5] = u[:, 1] * u[:, 5] / u[:, 0] - B1 * u[:, 3] / u[:, 0]
        f[:, 6] = (u[:, 6] + p_star) * (u[:, 1] / u[:, 0]) - B1 * (
            B1 * u[:, 1] + u[:, 2] * u[:, 4] + u[:, 3] * u[:, 5]
        ) / u[:, 0]

        return f

    def spectral_radius_x(self, u):
        rho = u[:, 0]
        u_x = u[:, 1] / rho
        p = self.pressure(u)
        A = 2.0 * p / rho
        B = (self.B1 ** 2 + u[:, 4] ** 2 + u[:, 5] ** 2) / rho
        cf = np.sqrt(
            0.5 * (A + B + np.sqrt((A + B) ** 2 - 4.0 * A * self.B1 ** 2 / rho))
        )
        return np.abs(u_x) + cf


###############
# 2D equations
###############

# Scalar equation
class Scalar2d(centpy.Equation2d):
    def initial_data(self):
        return np.sin(self.xx.T + 0.5) * np.cos(2 * self.xx.T + self.yy.T)

    def boundary_conditions(self, u):
        # x-boundary
        u[0] = u[-4]
        u[1] = u[-3]
        u[-2] = u[2]
        u[-1] = u[3]
        # y-boundary
        u[:, 0] = u[:, -4]
        u[:, 1] = u[:, -3]
        u[:, -2] = u[:, 2]
        u[:, -1] = u[:, 3]

    def flux_x(self, u):
        return np.sin(u)

    def flux_y(self, u):
        return 1.0 / 3.0 * u ** 3

    def spectral_radius_x(self, u):
        return np.abs(np.cos(u))

    def spectral_radius_y(self, u):
        return u ** 2


# Euler equation
class Euler2d(centpy.Equation2d):

    # Helper functions and definitions for the equation

    def pressure(self, u):
        return (self.gamma - 1.0) * (
            u[:, :, 3] - 0.5 * (u[:, :, 1] ** 2 + u[:, :, 2] ** 2) / u[:, :, 0]
        )

    def euler_data(self):
        gamma = self.gamma

        p_one = 1.5
        p_two = 0.3
        p_three = 0.029
        p_four = 0.3

        upper_right, upper_left, lower_right, lower_left = np.ones((4, 4))

        upper_right[0] = 1.5
        upper_right[1] = 0.0
        upper_right[2] = 0.0
        upper_right[3] = (
            p_one / (gamma - 1.0)
            + 0.5 * (upper_right[1] ** 2 + upper_right[2] ** 2) / upper_right[0]
        )

        upper_left[0] = 0.5323
        upper_left[1] = 1.206 * upper_left[0]
        upper_left[2] = 0.0
        upper_left[3] = (
            p_two / (gamma - 1.0)
            + 0.5 * (upper_left[1] ** 2 + upper_left[2] ** 2) / upper_left[0]
        )

        lower_right[0] = 0.5323
        lower_right[1] = 0.0
        lower_right[2] = 1.206 * lower_right[0]
        lower_right[3] = (
            p_four / (gamma - 1.0)
            + 0.5 * (lower_right[1] ** 2 + lower_right[2] ** 2) / lower_right[0]
        )

        lower_left[0] = 0.138
        lower_left[1] = 1.206 * lower_left[0]
        lower_left[2] = 1.206 * lower_left[0]
        lower_left[3] = (
            p_three / (gamma - 1.0)
            + 0.5 * (lower_left[1] ** 2 + lower_left[2] ** 2) / lower_left[0]
        )

        return upper_right, upper_left, lower_right, lower_left

    # Abstract class equation definitions

    def initial_data(self):
        u = np.empty((self.J + 4, self.K + 4, 4))
        midJ = int(self.J / 2) + 2
        midK = int(self.K / 2) + 2

        one_matrix = np.ones(u[midJ:, midK:].shape)
        upper_right, upper_left, lower_right, lower_left = self.euler_data()

        u[midJ:, midK:] = upper_right * one_matrix
        u[:midJ, midK:] = upper_left * one_matrix
        u[midJ:, :midK] = lower_right * one_matrix
        u[:midJ, :midK] = lower_left * one_matrix
        return u

    def boundary_conditions(self, u):

        upper_right, upper_left, lower_right, lower_left = self.euler_data()

        if self.odd:
            j = slice(1, -2)
            u[j, 0] = u[j, 1]
            u[j, -2] = u[j, -3]
            u[j, -1] = u[j, -3]

            u[0, j] = u[1, j]
            u[-2, j] = u[-3, j]
            u[-1, j] = u[-3, j]

            # one
            u[-2, -2] = upper_right
            u[-1, -2] = upper_right
            u[-2, -1] = upper_right
            u[-1, -1] = upper_right

            # two
            u[0, -2] = upper_left
            u[0, -1] = upper_left

            # three
            u[0, 0] = lower_left
            u[0, 1] = lower_left
            u[1, 0] = lower_left
            u[1, 1] = lower_left

            # four
            u[-2, 0] = lower_right
            u[-1, 0] = lower_right
            u[-2, 1] = lower_right
            u[-1, 1] = lower_right

        else:

            j = slice(2, -1)
            u[j, 0] = u[j, 2]
            u[j, 1] = u[j, 2]
            u[j, -1] = u[j, -2]

            u[0, j] = u[2, j]
            u[1, j] = u[2, j]
            u[-1, j] = u[-2, j]

            # one
            u[-1, -2] = upper_right
            u[-1, -1] = upper_right

            # two
            u[0, -2] = upper_left
            u[0, -1] = upper_left
            u[1, -2] = upper_left
            u[1, -1] = upper_left

            # three
            u[0, 0] = lower_left
            u[0, 1] = lower_left
            u[1, 0] = lower_left
            u[1, 1] = lower_left

            # four
            u[-1, 0] = lower_right
            u[-1, 1] = lower_right

    def flux_x(self, u):
        f = np.empty_like(u)

        p = self.pressure(u)

        f[:, :, 0] = u[:, :, 1]
        f[:, :, 1] = u[:, :, 1] ** 2 / u[:, :, 0] + p
        f[:, :, 2] = u[:, :, 1] * u[:, :, 2] / u[:, :, 0]
        f[:, :, 3] = (u[:, :, 3] + p) * u[:, :, 1] / u[:, :, 0]

        return f

    def flux_y(self, u):
        g = np.empty_like(u)

        p = self.pressure(u)

        g[:, :, 0] = u[:, :, 2]
        g[:, :, 1] = u[:, :, 1] * u[:, :, 2] / u[:, :, 0]
        g[:, :, 2] = u[:, :, 2] ** 2 / u[:, :, 0] + p
        g[:, :, 3] = (u[:, :, 3] + p) * u[:, :, 2] / u[:, :, 0]

        return g

    def spectral_radius_x(self, u):

        j0 = centpy._helpers.j0

        rho = u[j0, j0, 0]
        vx = u[j0, j0, 1] / rho
        vy = u[j0, j0, 2] / rho
        p = (self.gamma - 1.0) * (u[j0, j0, 3] - 0.5 * rho * (vx ** 2 + vy ** 2))
        c = np.sqrt(self.gamma * p / rho)
        return np.abs(vx) + c

    def spectral_radius_y(self, u):

        j0 = centpy._helpers.j0

        rho = u[j0, j0, 0]
        vx = u[j0, j0, 1] / rho
        vy = u[j0, j0, 2] / rho
        p = (self.gamma - 1.0) * (u[j0, j0, 3] - 0.5 * rho * (vx ** 2 + vy ** 2))
        c = np.sqrt(self.gamma * p / rho)
        return np.abs(vy) + c


# MHD equation
class MHD2d(centpy.Equation2d):

    # Helper functions for the equation

    def pressure(self, u):
        return (self.gamma - 1.0) * (
            u[:, :, 7]
            - 0.5 * (u[:, :, 1] ** 2 + u[:, :, 2] ** 2 + u[:, :, 3] ** 2) / u[:, :, 0]
            - 0.5 * (u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2)
        )

    def pressure_star(self, u):
        return self.pressure(u) + 0.5 * (
            u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2
        )

    def initial_data(self):
        u = np.zeros((self.J + 4, self.K + 4, 8))
        x = self.xx.T
        y = self.yy.T
        gamma = self.gamma
        dx = self.dx
        dy = self.dy

        u[:, :, 0] = gamma ** 2
        u[:, :, 1] = u[:, :, 0] / dy * (np.cos(y + 0.5 * dy) - np.cos(y - 0.5 * dy))
        u[:, :, 2] = 0.0
        u[:, :, 3] = -u[:, :, 0] / dx * (np.cos(x + 0.5 * dx) - np.cos(x - 0.5 * dx))
        u[:, :, 4] = 1.0 / dy * (np.cos(y + 0.5 * dy) - np.cos(y - 0.5 * dy))
        u[:, :, 5] = 0.0
        u[:, :, 6] = (
            -0.5 / dx * (np.cos(2.0 * (x + 0.5 * dx)) - np.cos(2.0 * (x - 0.5 * dx)))
        )

        I1 = (
            -0.125
            / dy
            * (u[:, :, 0] + 1.0)
            * (np.sin(2.0 * (y + 0.5 * dy)) - np.sin(2.0 * (y - 0.5 * dy)))
        )
        I2 = (
            -0.125
            / dx
            * u[:, :, 0]
            * (np.sin(2.0 * (x + 0.5 * dx)) - np.sin(2.0 * (x - 0.5 * dx)))
        )
        I3 = (
            -0.0625 / dx * (np.sin(4.0 * (x + 0.5 * dx)) - np.sin(4.0 * (x - 0.5 * dx)))
        )
        u[:, :, 7] = 3.0 + 0.5 * u[:, :, 0] + I1 + I2 + I3

        return u

    def boundary_conditions(self, u):  # periodic
        # x-boundary
        u[0] = u[-4]
        u[1] = u[-3]
        u[-2] = u[2]
        u[-1] = u[3]
        # y-boundary
        u[:, 0] = u[:, -4]
        u[:, 1] = u[:, -3]
        u[:, -2] = u[:, 2]
        u[:, -1] = u[:, 3]

    def flux_x(self, u):
        f = np.empty_like(u)

        p_star = self.pressure_star(u)

        f[:, :, 0] = u[:, :, 1]
        f[:, :, 1] = u[:, :, 1] ** 2 / u[:, :, 0] + p_star - u[:, :, 4] ** 2
        f[:, :, 2] = u[:, :, 1] * u[:, :, 2] / u[:, :, 0] - u[:, :, 4] * u[:, :, 5]
        f[:, :, 3] = u[:, :, 1] * u[:, :, 3] / u[:, :, 0] - u[:, :, 4] * u[:, :, 6]
        f[:, :, 4] = 0.0
        f[:, :, 5] = (
            u[:, :, 1] * u[:, :, 5] / u[:, :, 0] - u[:, :, 4] * u[:, :, 2] / u[:, :, 0]
        )
        f[:, :, 6] = (
            u[:, :, 1] * u[:, :, 6] / u[:, :, 0] - u[:, :, 4] * u[:, :, 3] / u[:, :, 0]
        )
        f[:, :, 7] = (u[:, :, 7] + p_star) * u[:, :, 1] / u[:, :, 0] - u[:, :, 4] * (
            u[:, :, 4] * u[:, :, 1] / u[:, :, 0]
            + u[:, :, 5] * u[:, :, 2] / u[:, :, 0]
            + u[:, :, 6] * u[:, :, 3] / u[:, :, 0]
        )

        return f

    def flux_y(self, u):
        g = np.empty_like(u)

        p_star = self.pressure_star(u)

        g[:, :, 0] = u[:, :, 3]
        g[:, :, 1] = u[:, :, 3] * u[:, :, 1] / u[:, :, 0] - u[:, :, 4] * u[:, :, 6]
        g[:, :, 2] = u[:, :, 3] * u[:, :, 2] / u[:, :, 0] - u[:, :, 5] * u[:, :, 6]
        g[:, :, 3] = u[:, :, 3] ** 2 / u[:, :, 0] + p_star - u[:, :, 6] ** 2

        g[:, :, 4] = (
            u[:, :, 3] * u[:, :, 4] / u[:, :, 0] - u[:, :, 6] * u[:, :, 1] / u[:, :, 0]
        )
        g[:, :, 5] = (
            u[:, :, 3] * u[:, :, 5] / u[:, :, 0] - u[:, :, 6] * u[:, :, 2] / u[:, :, 0]
        )
        g[:, :, 6] = 0.0
        g[:, :, 7] = (u[:, :, 7] + p_star) * u[:, :, 3] / u[:, :, 0] - u[:, :, 6] * (
            u[:, :, 4] * u[:, :, 1] / u[:, :, 0]
            + u[:, :, 5] * u[:, :, 2] / u[:, :, 0]
            + u[:, :, 6] * u[:, :, 3] / u[:, :, 0]
        )

        return g

    def spectral_radius_x(self, u):
        rho = u[:, :, 0]
        vx = u[:, :, 1] / rho
        vy = u[:, :, 2] / rho
        vz = u[:, :, 3] / rho
        p = (self.gamma - 1.0) * (
            u[:, :, 7]
            - 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
            - 0.5 * (u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2)
        )
        A = self.gamma * p / rho
        B = (u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2) / rho
        cfx = np.sqrt(
            0.5 * (A + B + np.sqrt((A + B) ** 2 - 4 * A * u[:, :, 4] ** 2 / rho))
        )
        # cfy = np.sqrt(
        #     0.5 * (A + B + np.sqrt((A + B) ** 2 - 4 * A * u[:, :, 6] ** 2 / rho))
        # )

        return np.abs(vx) + cfx

    def spectral_radius_y(self, u):
        rho = u[:, :, 0]
        vx = u[:, :, 1] / rho
        vy = u[:, :, 2] / rho
        vz = u[:, :, 3] / rho
        p = (self.gamma - 1.0) * (
            u[:, :, 7]
            - 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
            - 0.5 * (u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2)
        )
        A = self.gamma * p / rho
        B = (u[:, :, 4] ** 2 + u[:, :, 5] ** 2 + u[:, :, 6] ** 2) / rho
        # cfx = np.sqrt(
        #     0.5 * (A + B + np.sqrt((A + B) ** 2 - 4 * A * u[:, :, 4] ** 2 / rho))
        # )
        cfy = np.sqrt(
            0.5 * (A + B + np.sqrt((A + B) ** 2 - 4 * A * u[:, :, 6] ** 2 / rho))
        )

        return np.abs(vy) + cfy
