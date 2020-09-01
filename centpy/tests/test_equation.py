import pytest
from centpy.equations import Equation1d
from centpy.parameters import Pars1d


class Scalar(Equation1d):
    def initial_data(self):
        return [20, 12]

    def boundary_conditions(self, u):
        u[0] = 10

    def flux_x(self, u):
        return 0.5 * u

    def spectral_radius_x(self, u):
        return 0.1 * u


@pytest.fixture
def eqn():
    return Scalar(Pars1d())


def test_equation1d(eqn):
    tmp_u = [100, 200]
    eqn.boundary_conditions(tmp_u)
    assert eqn.initial_data() == [20, 12]
    assert tmp_u == [10, 200]
    assert eqn.flux_x(tmp_u[0]) == 5
    assert eqn.spectral_radius_x(tmp_u[1]) == 20
