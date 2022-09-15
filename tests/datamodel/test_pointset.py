import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPointSet
from vtkmodules.vtkFiltersSources import vtkSphereSource


def test_pointset_center_of_mass(cube):
    pset = vtkPointSet()
    pset.points = cube.points
    assert np.allclose(pset.center_of_mass(), (0, 0, 0))


def test_pointset_points_to_double():
    source = vtkSphereSource()
    source.SetRadius(1.0)
    source.SetCenter((1, 1, 1))
    source.Update()
    mesh = source.GetOutput()
    assert mesh.points.dtype == np.dtype("float32")
    mesh.points_to_double()
    assert mesh.points.dtype == np.dtype("float64")


def test_pointset_cast_to_polydata(sphere):
    pset = vtkPointSet()
    pset.points = sphere.points
    poly = pset.cast_to_polydata()
    assert poly.n_points == sphere.n_points
    assert poly.n_cells == poly.n_points
