import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPolyData


def test_poly_eq(sphere):
    sphere.clear_data()
    sphere.point_data["data0"] = np.zeros(sphere.n_points)
    sphere.point_data["data1"] = np.arange(sphere.n_points)

    copy = sphere.copy(deep=True)
    assert sphere == copy

    copy.faces = [3, 0, 1, 2]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.field_data["new"] = [1]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_data["new"] = range(sphere.n_points)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.cell_data["new"] = range(sphere.n_cells)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_data.active_scalars_name = "data0"
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.lines = [2, 0, 1]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.verts = [1, 0]
    assert sphere != copy


def test_poly_verts():
    points = np.random.random((5, 3))
    poly = vtkPolyData()
    poly.points = points
    poly.make_vertex_cells()
    assert np.allclose(poly.verts, np.array([1, 0, 1, 1, 1, 2, 1, 3, 1, 4]))


def test_poly_lines():
    points = np.random.random((3, 3))
    lines = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3]])
    poly = vtkPolyData()
    poly.points = points
    poly.lines = lines.ravel()
    for i in range(poly.n_cells):
        assert np.allclose(lines[i, 1:], poly.cell_point_ids(i))


def test_poly_faces(cube):
    assert cube.n_faces == 6
    assert cube.faces.size / 5 == 6
    faces = np.reshape(cube.faces, (-1, 5))
    for i, face in enumerate(faces):
        assert np.allclose(face[1:], cube.cell_point_ids(i))
