import numpy as np
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkStructuredGrid

from vtk_override.datamodel.rectilinear import RectilinearGrid


def test_rectilinear_init():
    x = [0, 1]
    y = [0, 1]
    z = [0, 1]
    grid = RectilinearGrid(x, y, z)
    assert isinstance(grid.dimensions, tuple)
    assert np.allclose(grid.x, x)


def test_rectilinear_init_no_dimensions():
    grid = RectilinearGrid()
    assert isinstance(grid.dimensions, tuple)


def test_rectilinear_meshgrid():
    x = [0, 1]
    y = [0, 1]
    z = [0, 1]
    grid = RectilinearGrid(x, y, z)
    # todo
    # grid_mesh = grid.meshgrid
    # np_mesh = np.meshgrid(x, y, z)
    # assert (grid_mesh == np_mesh).any()


def test_rectlinear_points():
    xrng = np.arange(-10, 10, 10, dtype=float)
    yrng = np.arange(-10, 10, 10, dtype=float)
    zrng = np.arange(-10, 10, 10, dtype=float)
    grid = RectilinearGrid(xrng, yrng, zrng)
    points = [
        [-10.0, -10.0, -10.0],
        [0.0, -10.0, -10.0],
        [-10.0, 0.0, -10.0],
        [0.0, 0.0, -10.0],
        [-10.0, -10.0, 0.0],
        [0.0, -10.0, 0.0],
        [-10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    # rectilinear.py:21
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    assert np.allclose(grid.points, points)


def test_cast_rectilinear_to_structured_grid():
    grid = RectilinearGrid()
    structured_grid = grid.cast_to_structured_grid()
    assert isinstance(structured_grid, vtkStructuredGrid)
    assert structured_grid.n_points == grid.n_points
    assert structured_grid.n_cells == grid.n_cells
