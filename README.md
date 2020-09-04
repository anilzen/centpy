# centpy
Central schemes for conservation laws in Python.

The schemes are translated into Python from [CentPack](https://home.cscamm.umd.edu/centpack/) written by [Jorge Balbás](http://www.csun.edu/~jb715473/index.htm) and [Eitan Tadmor](https://home.cscamm.umd.edu/people/faculty/tadmor/).

## Usage

Centpy provides to the user three main classes for parameters, equations, and solvers. Examples of instances for parameters and equations are in [`tests/example_parameters.py`](centpy/tests/example_parameters.py) and [`tests/example_equations.py`](centpy/tests/example_equations.py).

The numerical solution of a one-dimensional Burgers equation is discussed below.

### Parameters
The parameter classes are simple [data classes](https://docs.python.org/3/library/dataclasses.html) without methods: `Pars1d` and `Pars1d` defined in [`parameters.py`](centpy/parameters.py). Each attribute has a default 
variable, but it is recommended that all attributes are set explicitly. The attributes are:

| Attribute | Description | 
| --------- | ----------- |
| `x_init`  | left grid point |
| `x_final` | right grid point|
| `t_final` | evolution time |
| `dt_out`  | time step of storage |
| `J`       | number of interior grid points |
| `cfl`     | CFL number |
| `scheme`  | solver scheme (`fd2`, `sd2`, or `sd3`) |

An instance of the parameter class can be created as follows. 

```
pars_burgers1d = centpy.Pars1d(
    x_init=0.0,
    x_final=2.0 * np.pi,
    t_final=10,
    dt_out=0.05,
    J=400,
    cfl=0.75,
    scheme="sd3")
```
Note that the parameter data class does not have a member for the time step `dt`, because it is calculated dynamically during the solution of the equation based on the CFL number and maximum spectral radius. 

### Equations

The equations are [abstract base classes](https://docs.python.org/3/library/abc.html) which require methods for setting initial data, boundary conditions, fluxes, and spectral radius. Additional helper methods and parameters can be added depending on the problem. The equations class inherits all attributes of the parameters class.  The space-time grid is constructed in this step based on the parameters. The Burgers equation class is defined below.

```
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
```
The boundary conditions are periodic, so the data on the ghost points are copied from the interior points on the opposite end. 

### Solution

There are two solver classes: `Solver1d` and `Solver2d` defined in [`solver1d.py`](centpy/solver1d.py) and [`solver2d.py`](centpy/solver2d.py) respectively. To construct the solution, we create an instance of the `Burgers1d` class with the parameters, and give the equation instance as input to the solver class. 

```
eqn_burgers1d = Burgers1d(pars_burgers1d)
soln_burgers = centpy.Solver1d(eqn_burgers1d)
soln_burgers.solve()
```

After the solver step, the instance `soln_burgers` includes the solution array `u_n`. Depending on the shape of the array, plots and animations can be easily constructed. Examples are given in the animations notebook `tests/animations.ipynb`.

The options for the central solver are `fd2` for second order fully-discrete method, `sd2` for second order semi-discrete method, and `sd3` for third order semi-discrete method. Information about these solvers is given at the appendix of the [CentPack User Guide](https://home.cscamm.umd.edu/centpack/documentation/CP_user_guide.pdf).

LaTeX formulas and animations for the examples are given in the Jupyter notebook `tests/animations.ipynb`. 
