import vtk

from vtk_override.utils import override


def test_override_call():
    class Foo(vtk.vtkPolyData):
        pass

    assert not isinstance(vtk.vtkPolyData(), Foo)
    override(vtk.vtkPolyData, Foo)
    assert isinstance(vtk.vtkPolyData(), Foo)
    override(vtk.vtkPolyData, None)
    assert not isinstance(vtk.vtkPolyData(), Foo)


def test_override_decorator():
    @override(vtk.vtkPolyData)
    class Foo(vtk.vtkPolyData):
        pass

    assert isinstance(vtk.vtkPolyData(), Foo)
    override(vtk.vtkPolyData, None)
    assert not isinstance(vtk.vtkPolyData(), Foo)
