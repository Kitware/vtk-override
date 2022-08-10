"""vtkDataObject subclasses.

https://vtk.org/doc/nightly/html/classvtkDataObject.html
"""
# flake8: noqa: F401
import vtk  # TODO: https://gitlab.kitware.com/vtk/vtk/-/issues/18594

from vtk_override.datamodel.dataset import DataSet, DataSetBase
from vtk_override.datamodel.image import ImageData, ImageDataBase, StructuredPoints, UniformGrid
from vtk_override.datamodel.pointset import *
from vtk_override.datamodel.rectilinear import RectilinearGrid
