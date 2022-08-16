import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPointSet,
    vtkPolyData,
    vtkUnstructuredGrid,
)

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


def test_copy_attributes(wavelet):
    raise NotImplementedError


def test_cast_to_unstructured_grid(sphere):
    casted = sphere.cast_to_unstructured_grid()
    assert isinstance(casted, vtkUnstructuredGrid)
    assert np.allclose(sphere.points, casted.points)


def test_cast_to_pointset(sphere):
    casted = sphere.cast_to_pointset()
    assert isinstance(casted, vtkPointSet)
    assert np.allclose(sphere.points, casted.points)


def test_cell_points(cube):
    cube.cell_points(0)
    # AttributeError: 'NoneType' object has no attribute 'cell_points


def test_cell_bounds(cube):
    cube.cell_points(0)
    # AttributeError: 'NoneType' object has no attribute 'cell_points


def test_cell_type():
    obj = vtkPointSet()
    cell_type = obj.cell_type(1)
    assert cell_type == 0  # empty cell


def test_cell_type_non_empty():
    obj = vtkPointSet()
    cell_type = obj.cell_type(1)
    # todo - insert cell?
    assert cell_type == 0  # empty cell


def test_cell_point_ids():
    obj = vtkImageData()
    point_ids = obj.cell_type(0)
    assert point_ids == 0
