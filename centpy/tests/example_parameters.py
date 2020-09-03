# Setup parameters to be used in various examples

from path import *
import numpy as np
import centpy

pars_burgers1d = centpy.Pars1d(
    x_init=0.0,
    x_final=2.0 * np.pi,
    t_final=10,
    dt_out=0.05,
    J=400,
    cfl=0.75,
    scheme="sd3",
)
pars_euler1d = centpy.Pars1d(
    x_init=0.0, x_final=1.0, t_final=0.3, dt_out=0.004, J=400, cfl=0.75, scheme="sd2"
)
pars_euler1d.gamma = 1.4
pars_mhd1d = centpy.Pars1d(
    x_init=-1.0, x_final=1.0, t_final=0.2, dt_out=0.002, J=400, cfl=0.475, scheme="fd2"
)
pars_mhd1d.B1 = 0.75
pars_scalar2d = centpy.Pars2d(
    x_init=0.0,
    x_final=2.0 * np.pi,
    y_init=0.0,
    y_final=2.0 * np.pi,
    J=144,
    K=144,
    t_final=6.0,
    dt_out=0.1,
    cfl=0.9,
    scheme="sd3",
)
pars_euler2d = centpy.Pars2d(
    x_init=0.0,
    x_final=1.0,
    y_init=0.0,
    y_final=1.0,
    J=200,
    K=200,
    t_final=0.4,
    dt_out=0.005,
    cfl=0.475,
    scheme="fd2",
)
pars_euler2d.gamma = 1.4
pars_mhd2d = centpy.Pars2d(
    x_init=0.0,
    x_final=2.0 * np.pi,
    y_init=0.0,
    y_final=2.0 * np.pi,
    J=144,
    K=144,
    t_final=3.0,
    dt_out=0.05,
    cfl=0.75,
    scheme="sd2",
)
pars_mhd2d.gamma = 5.0 / 3
