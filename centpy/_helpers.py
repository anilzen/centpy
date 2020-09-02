import numpy as np

# Parameters for the solvers
alpha = 1.4  # for the minmod limiter
eps = 0.000001  # when using sd2

# Index definitions for convenience
j0 = slice(2, -2)
jp = slice(3, -1)
jm = slice(1, -3)

####################
# Limiters (minmod, va)
####################

# 1d


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

# 2d

# Reuse code
def minmod_prime_x(u):
    return minmod_prime(u[:, 1:-1])
    # return minmod3(
    #     alpha * (u[1:-1, 1:-1] - u[:-2, 1:-1]),
    #     0.5 * (u[2:, 1:-1] - u[:-2, 1:-1]),
    #     alpha * (u[2:, 1:-1] - u[1:-1, 1:-1]),
    # )


def minmod_prime_y(u):
    return minmod3(
        alpha * (u[1:-1, 1:-1] - u[1:-1, :-2]),
        0.5 * (u[1:-1, 2:] - u[1:-1, :-2]),
        alpha * (u[1:-1, 2:] - u[1:-1, 1:-1]),
    )


# Reuse code
def va_prime_x(u):
    return va_prime(u[:, 1:-1])


def va_prime_y(u):
    return va(u[1:-1, 2:] - u[1:-1, 1:-1], u[1:-1, 1:-1] - u[1:-1, :-2])


limiter_x = va_prime_x
limiter_y = va_prime_y


####################
# p coefficients
####################

# 1d


def p_coefs(u):
    pl0 = u[j0]
    pl1 = u[j0] - u[jm]
    pr0 = u[j0]
    pr1 = u[jp] - u[j0]
    pc0 = pl0 - (1.0 / 12.0) * (pr1 - pl1)
    pc1 = 0.5 * (pl1 + pr1)
    pc2 = pr1 - pl1
    return pl0, pl1, pr0, pr1, pc0, pc1, pc2


# 2d


def px_coefs(u):
    pl0 = u[j0, j0]
    pl1 = u[j0, j0] - u[jm, j0]
    pr0 = pl0.copy()
    pr1 = u[jp, j0] - u[j0, j0]

    pcx0 = pl0 - 1.0 / 12.0 * (
        u[jp, j0] + u[j0, jp] - 4.0 * u[j0, j0] + u[jm, j0] + u[j0, jm]
    )
    pcx1 = 0.5 * (pl1 + pr1)
    pcx2 = pr1 - pl1
    return pl0, pl1, pr0, pr1, pcx0, pcx1, pcx2


def py_coefs(u):
    pb0 = u[j0, j0]
    pb1 = u[j0, j0] - u[j0, jm]
    pt0 = pb0.copy()
    pt1 = u[j0, jp] - u[j0, j0]

    pcy0 = pb0 - 1.0 / 12.0 * (
        u[jp, j0] + u[j0, jp] - 4.0 * u[j0, j0] + u[jm, j0] + u[j0, jm]
    )
    pcy1 = 0.5 * (pb1 + pt1)
    pcy2 = pt1 - pb1
    return pb0, pb1, pt0, pt1, pcy0, pcy1, pcy2


def pdx_coefs(u):
    pl0 = u[j0, j0].copy()
    pl1 = u[j0, j0] - u[jm, jm]
    pr0 = pl0.copy()
    pr1 = u[jp, jp] - u[j0, j0]

    pcx0 = pl0 - 1.0 / 12.0 * (
        u[jp, jp] + u[jm, jp] - 4.0 * u[j0, j0] + u[jm, jm] + u[jp, jm]
    )
    pcx1 = 0.5 * (pl1 + pr1)
    pcx2 = pr1 - pl1
    return pl0, pl1, pr0, pr1, pcx0, pcx1, pcx2


def pdy_coefs(u):
    pb0 = u[j0, j0].copy()
    pb1 = u[j0, j0] - u[jp, jm]
    pt0 = pb0.copy()
    pt1 = u[jm, jp] - u[j0, j0]

    pcy0 = pb0 - 1.0 / 12.0 * (
        u[jp, jp] + u[jm, jp] - 4.0 * u[j0, j0] + u[jm, jm] + u[jp, jm]
    )
    pcy1 = 0.5 * (pb1 + pt1)
    pcy2 = pt1 - pb1
    return pb0, pb1, pt0, pt1, pcy0, pcy1, pcy2

