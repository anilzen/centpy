from centpy.parameters import Pars1d, Pars2d
import pytest

# @pytest.fixture
# pars = Pars1d(x_init=0.0, x_final=1.0, t_final=10, dt_out=0.05, J=400, cfl=0.75)


def test_pars1d():
    pars = Pars1d(x_init=0.0, x_final=1.0, t_final=10, dt_out=0.05, J=400, cfl=0.75)
    assert pars.x_init == 0.0
    assert pars.x_final == 1.0
    assert pars.t_final == 10.0
    assert pars.dt_out == 0.05
    assert pars.J == 400
    assert pars.cfl == 0.75
    assert pars.scheme == "sd3"


def test_pars2d():
    pars = Pars2d(y_init=0.0, y_final=1.0, J=144, K=300, scheme="fd2")
    assert pars.y_init == 0.0
    assert pars.y_final == 1.0
    assert pars.J == 144
    assert pars.K == 300
    assert pars.scheme == "fd2"


# def test_invalid_scheme():
#     with pytest.raises(ValueError) as exp:
#         pars = Pars1d(scheme="sd4")
#     assert str(exp.value) == "Invalid scheme entry."
