# Generate test data to be used in pytest
from definitions import *

for pars, eqn, name in zip(pars_list, eqn_list, name_list):
    pars.t_final = FINAL_T
    pars.dt_out = OUT_DT
    u_n = solve(eqn(pars))
    np.save("data/" + name + ".npy", u_n)
