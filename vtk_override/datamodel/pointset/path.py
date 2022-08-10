from vtkmodules.vtkCommonDataModel import vtkPath

from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override


@override(vtkPath)
class Path(PointSetBase, vtkPath):
    pass
