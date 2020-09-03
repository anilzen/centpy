# Generate test data to be used in pytest
from example_parameters import *
from example_equations import *


def save_soln(eqn, name):
    if hasattr(eqn, "y"):
        soln = centpy.Solver2d(eqn)
    else:
        soln = centpy.Solver1d(eqn)
    soln.solve()
    np.save("data/" + name + ".npy", soln.u_n)


# Five timesteps for each equation
OUT_DT = 0.001
FINAL_T = 0.005

# Set t_final and dt_out for parameter instances
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

for pars, eqn, name in zip(pars_list, eqn_list, name_list):
    pars.t_final = FINAL_T
    pars.out_dt = OUT_DT
    save_soln(eqn(pars), name)
