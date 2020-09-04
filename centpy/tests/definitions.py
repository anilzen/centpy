# Generate test data to be used in pytest
from .example_parameters import *
from .example_equations import *

# Solve equation and return solution depending on dimension
def solve(eqn):
    if hasattr(eqn, "y"):
        soln = centpy.Solver2d(eqn)
    else:
        soln = centpy.Solver1d(eqn)
    soln.solve()
    return soln.u_n


# Five timesteps for each equation
OUT_DT = 0.001
FINAL_T = 0.005

# Lists for parameters, equations, and filenames without extensions
pars_list = [
    pars_burgers1d,
    pars_euler1d,
    pars_mhd1d,
    pars_scalar2d,
    pars_euler2d,
    pars_mhd2d,
]
eqn_list = [Burgers1d, Euler1d, MHD1d, Scalar2d, Euler2d, MHD2d]
name_list = ["burgers1d", "euler1d", "mhd1d", "scalar2d", "euler2d", "mhd2d"]
