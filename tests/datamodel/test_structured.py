import numpy as np
import pytest
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid


@pytest.fixture
def structured_points():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    source = np.empty((x.size, 3), x.dtype)
    source[:, 0] = x.ravel("F")
    source[:, 1] = y.ravel("F")
    source[:, 2] = z.ravel("F")
    return source, (*x.shape, 1)


def test_no_copy_structured_mesh_points_setter(structured_points):
    source, dims = structured_points
    mesh = vtkStructuredGrid()
    mesh.points = source
    mesh.dimensions = dims
    pts = mesh.points
    pts /= 2
    assert np.array_equal(mesh.points, pts)
    assert np.may_share_memory(mesh.points, pts)
    assert np.array_equal(mesh.points, source)
    assert np.may_share_memory(mesh.points, source)
