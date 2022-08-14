import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

# def test_shallow_copy(sphere):
#     # Case 1
#     points = vtkPoints()
#     points.InsertNextPoint(0.0, 0.0, 0.0)
#     points.InsertNextPoint(1.0, 0.0, 0.0)
#     points.InsertNextPoint(2.0, 0.0, 0.0)
#     original = vtkPolyData()
#     original.SetPoints(points)
#     wrapped = pyvista.PolyData(original, deep=False)
#     wrapped.points[:] = 2.8
#     orig_points = vtk_to_numpy(original.GetPoints().GetData())
#     assert np.allclose(orig_points, wrapped.points)
#     # Case 2
#     original = vtk.vtkPolyData()
#     wrapped = pyvista.PolyData(original, deep=False)
#     wrapped.points = np.random.rand(5, 3)
#     orig_points = vtk_to_numpy(original.GetPoints().GetData())
#     assert np.allclose(orig_points, wrapped.points)


def test_deep_copy(sphere):
    copy = sphere.copy(deep=False)
    assert copy.memory_address != sphere.memory_address
    assert copy.actual_memory_size == sphere.actual_memory_size


def test_copy():
    raise NotImplementedError


def test__eq__(cube):
    raise NotImplementedError


def test_copy_structure(sphere):
    copy = vtkPolyData()
    copy.copy_structure(sphere)
    # todo - add assertions
    assert copy.bounds == sphere.bounds


def test_copy_attributes(wavelet):
    raise NotImplementedError


def test_cast_to_unstructured_grid(sphere):
    raise NotImplementedError


def test_cast_to_pointset(sphere):
    raise NotImplementedError


def test_cell_points(cube):
    raise NotImplementedError


def test_cell_bounds(cube):
    raise NotImplementedError


def test_cell_type():
    raise NotImplementedError


def test_cell_point_ids(cube):
    raise NotImplementedError
