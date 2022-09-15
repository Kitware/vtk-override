import numpy as np
import pytest
from vtkmodules.vtkCommonDataModel import (
    VTK_QUAD,
    VTK_TRIANGLE,
    vtkImageData,
    vtkPointSet,
    vtkPolyData,
    vtkUnstructuredGrid,
)


def test_shallow_copy():
    # Case 1
    original = vtkPolyData()
    original.points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    copy = original.copy(deep=False)
    copy.points[:] = 2.8
    assert np.allclose(original.points, copy.points)
    # Case 2
    original = vtkPolyData()
    ###
    # TODO: remove this and figure out how to rectify original's `GetPoints()->None`
    original.points = np.random.rand(5, 3)
    ###
    copy = original.copy(deep=False)
    copy.points = np.random.rand(5, 3)
    assert np.allclose(original.points, copy.points)


def test_deep_copy(sphere):
    copy = sphere.copy(deep=False)
    assert copy.memory_address != sphere.memory_address
    assert copy.actual_memory_size == sphere.actual_memory_size


def test_copy_structure_poly(sphere):
    copy = vtkPolyData()
    copy.copy_structure(sphere)
    assert copy.bounds == sphere.bounds
    assert copy.center == sphere.center
    assert np.allclose(copy.points, sphere.points)
    assert np.allclose(copy.faces, sphere.faces)


def test_copy_structure_image(wavelet):
    copy = vtkImageData()
    copy.copy_structure(wavelet)
    assert copy.bounds == wavelet.bounds
    assert copy.center == wavelet.center
    assert copy.dimensions == wavelet.dimensions
    assert copy.spacing == wavelet.spacing
    assert copy.origin == wavelet.origin


def test_cast_to_unstructured_grid(sphere):
    casted = sphere.cast_to_unstructured_grid()
    assert isinstance(casted, vtkUnstructuredGrid)
    assert np.allclose(sphere.points, casted.points)


def test_cast_to_pointset(sphere):
    casted = sphere.cast_to_pointset()
    assert isinstance(casted, vtkPointSet)
    assert np.allclose(sphere.points, casted.points)


def test_cell_points(cube):
    points = cube.cell_points(0)
    assert len(points) == 4


def test_cell_bounds(cube):
    bounds = cube.cell_bounds(0)
    assert len(bounds) == 6


def test_cell_type(cube, sphere):
    assert cube.cell_type(0) == VTK_QUAD
    assert sphere.cell_type(0) == VTK_TRIANGLE


def test_cell_point_ids(cube):
    point_ids = cube.cell_point_ids(0)
    assert len(point_ids) == 4
    points = cube.cell_points(0)
    verify = cube.points[point_ids]
    assert np.allclose(points, verify)
