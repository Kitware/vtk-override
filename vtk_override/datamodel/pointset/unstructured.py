from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkUnstructuredGridBase

from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override


class UnstructuredGridBaseBase(PointSetBase):
    pass


@override(vtkUnstructuredGridBase)
class UnstructuredGridBase(UnstructuredGridBaseBase, vtkUnstructuredGridBase):
    pass


@override(vtkUnstructuredGrid)
class UnstructuredGrid(UnstructuredGridBaseBase, vtkUnstructuredGrid):
    pass


# @override(vtkMappedUnstructuredGrid)
# class MappedUnstructuredGrid(UnstructuredGridBaseBase, vtkMappedUnstructuredGrid):
#     pass
