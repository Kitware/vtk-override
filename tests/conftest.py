import pytest

# Import vtk_override so that it's classes override VTK
import vtk_override  # noqa
from vtk_override.utils.sources import Cube, Sphere, Wavelet


@pytest.fixture
def sphere():
    return Sphere()


@pytest.fixture
def cube():
    return Cube()


@pytest.fixture
def wavelet():
    return Wavelet()
