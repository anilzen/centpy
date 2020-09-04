from .definitions import *


def test_solutions():
    for pars, eqn, name in zip(pars_list, eqn_list, name_list):
        # Generate solution
        pars.t_final = FINAL_T
        pars.dt_out = OUT_DT
        u_n = solve(eqn(pars))
        # Load solution
        z_n = np.load("data/" + name + ".npy")
        # Compare
        assert not (z_n - u_n).any()
