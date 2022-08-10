"""Wrapping of vtkDataSet.

vtkDataSet
 |- vtkRectilinearGrid
 |- vtkImageData
 |- vtkPointSet
 ...

"""
from typing import List, Tuple

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkPointSet, vtkUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkAppendFilter

from vtk_override.datamodel.object import DataObjectBase
from vtk_override.utils import override, vtk_ndarray, vtk_points
from vtk_override.utils._typing import Vector
from vtk_override.utils.arrays import coerce_pointslike_arg, vtk_to_numpy


class DataSetBase(DataObjectBase):
    def __eq__(self, other):
        """Test equivalency between data objects."""
        if not isinstance(self, type(other)):
            return False

        if self is other:
            return True

        # these attrs use numpy.array_equal
        equal_attrs = [
            "verts",  # DataObject
            "points",  # DataObject
            "lines",  # DataObject
            "faces",  # DataObject
            "cells",  # UnstructuredGrid
            "celltypes",
        ]  # UnstructuredGrid
        for attr in equal_attrs:
            if hasattr(self, attr):
                if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                    return False

        # these attrs can be directly compared
        attrs = ["field_data", "point_data", "cell_data"]
        for attr in attrs:
            if hasattr(self, attr):
                if getattr(self, attr) != getattr(other, attr):
                    return False

        return True

    def copy_structure(self, dataset: vtkDataSet):
        """Copy the structure (geometry and topology) of the input dataset object.

        Parameters
        ----------
        dataset : vtk.vtkDataSet
            Dataset to copy the geometry and topology from.

        Examples
        --------
        >>> import vtk
        >>> source = vtk.vtkImageData()
        >>> source.dimensions = (10, 10, 5)
        >>> target = vtk.vtkImageData()
        >>> target.copy_structure(source)

        """
        self.CopyStructure(dataset)

    def copy_attributes(self, dataset: vtkDataSet):
        """Copy the data attributes of the input dataset object.

        Parameters
        ----------
        dataset : vtk.vtkDataSet
            Dataset to copy the data attributes from.

        Examples
        --------
        >>> import vtk
        >>> source = vtk.vtkImageData(dimensions=(10, 10, 5))
        >>> source = source.compute_cell_sizes()
        >>> target = vtk.vtkImageData(dimensions=(10, 10, 5))
        >>> target.copy_attributes(source)

        """
        self.CopyAttributes(dataset)

    def cast_to_unstructured_grid(self) -> vtkUnstructuredGrid:
        """Get a new representation of this object as a :class:`vtk.vtkUnstructuredGrid`.

        Returns
        -------
        vtk.vtkUnstructuredGrid
            Dataset cast into a :class:`vtk.vtkUnstructuredGrid`.

        Examples
        --------
        Cast a :class:`vtk.vtkPolyData` to a
        :class:`vtk.vtkUnstructuredGrid`.

        >>> import vtk
        >>> from vtk_override.utils import Sphere
        >>> mesh = Sphere()
        >>> type(mesh)
        <class 'vtk_override.datamodel.pointset.poly.PolyData'>
        >>> grid = mesh.cast_to_unstructured_grid()
        >>> type(grid)
        <class 'vtk_override.datamodel.pointset.unstructured.UnstructuredGrid'>

        """
        alg = vtkAppendFilter()
        alg.AddInputData(self)
        alg.Update()
        return alg.GetOutput()

    def cast_to_pointset(self, deep: bool = False) -> vtkPointSet:
        """Get a new representation of this object as a :class:`vtk.vtkPointSet`.

        Parameters
        ----------
        deep : bool, optional
            When ``True`` makes a full copy of the object.  When ``False``,
            performs a shallow copy where the points and data arrays are
            references to the original object.

        Returns
        -------
        vtk.vtkPointSet
            Dataset cast into a :class:`vtk.vtkPointSet`.

        Examples
        --------
        >>> from vtk_override.utils import Sphere
        >>> mesh = Sphere()
        >>> pointset = mesh.cast_to_pointset()
        >>> type(pointset)
        <class 'vtk_override.datamodel.pointset.points.PointSet'>

        """
        pset = vtkPointSet()
        pset.SetPoints(self.GetPoints())
        pset.GetPointData().ShallowCopy(self.GetPointData())
        if deep:
            return pset.copy(deep=True)
        return pset

    @property
    def points(self) -> vtk_ndarray:
        """Return a reference to the points as a numpy object.

        Examples
        --------
        Create a mesh and return the points of the mesh as a numpy
        array.

        >>> import vtk
        >>> from vtk_override.utils import Cube
        >>> mesh = Cube()
        >>> points = cube.points
        >>> points
        vtk_ndarray([[-0.5, -0.5, -0.5],
                         [-0.5, -0.5,  0.5],
                         [-0.5,  0.5,  0.5],
                         [-0.5,  0.5, -0.5],
                         [ 0.5, -0.5, -0.5],
                         [ 0.5,  0.5, -0.5],
                         [ 0.5,  0.5,  0.5],
                         [ 0.5, -0.5,  0.5]], dtype=float32)

        Shift these points in the z direction and show that their
        position is reflected in the mesh points.

        >>> points[:, 2] += 1
        >>> cube.points
        vtk_ndarray([[-0.5, -0.5,  0.5],
                         [-0.5, -0.5,  1.5],
                         [-0.5,  0.5,  1.5],
                         [-0.5,  0.5,  0.5],
                         [ 0.5, -0.5,  0.5],
                         [ 0.5,  0.5,  0.5],
                         [ 0.5,  0.5,  1.5],
                         [ 0.5, -0.5,  1.5]], dtype=float32)

        You can also update the points in-place:

        >>> cube.points[...] = 2*points
        >>> cube.points
        vtk_ndarray([[-1., -1.,  1.],
                         [-1., -1.,  3.],
                         [-1.,  1.,  3.],
                         [-1.,  1.,  1.],
                         [ 1., -1.,  1.],
                         [ 1.,  1.,  1.],
                         [ 1.,  1.,  3.],
                         [ 1., -1.,  3.]], dtype=float32)

        """
        _points = self.GetPoints()
        try:
            _points = _points.GetData()
        except AttributeError:
            # create an empty array
            pts = vtk_points(np.empty((0, 3)), False)
            self.SetPoints(pts)
            _points = self.GetPoints().GetData()
        return vtk_ndarray(_points, dataset=self)

    @points.setter
    def points(self, points):  # Union[VectorArray, NumericArray, vtkPoints]
        pdata = self.GetPoints()
        if isinstance(points, vtk_ndarray):
            # simply set the underlying data
            if points.VTKObject is not None and pdata is not None:
                pdata.SetData(points.VTKObject)
                pdata.Modified()
                self.Modified()
                return
        # directly set the data if vtk object
        if isinstance(points, vtkPoints):
            self.SetPoints(points)
            if pdata is not None:
                pdata.Modified()
            self.Modified()
            return
        # otherwise, wrap and use the array
        points = coerce_pointslike_arg(points, copy=False)
        pts = vtk_points(points, False)
        if not pdata:
            self.SetPoints(pts)
        else:
            pdata.SetData(pts.GetData())
        self.GetPoints().Modified()
        self.Modified()

    @property
    def point_data(self):
        pd = super().GetPointData()
        pd.dataset = self
        # TODO: temporary hack
        pd.dataset.VTKObject = pd.dataset
        pd.association = self.POINT
        return pd

    def clear_point_data():
        raise NotImplementedError

    @property
    def cell_data(self):
        cd = super().GetCellData()
        cd.dataset = self
        cd.association = self.CELL
        return cd

    def clear_cell_data():
        raise NotImplementedError

    @property
    def n_arrays(self) -> int:
        """Return the number of arrays present in the dataset."""
        n = self.GetPointData().GetNumberOfArrays()
        n += self.GetCellData().GetNumberOfArrays()
        n += self.GetFieldData().GetNumberOfArrays()
        return n

    @property
    def n_points(self) -> int:
        """Return the number of points in the entire dataset.

        Examples
        --------
        Create a mesh and return the number of points in the
        mesh.

        >>> import vtk
        >>> from vtk_override.utils import Cube
        >>> mesh = Cube()
        >>> cube.n_points
        8

        """
        return self.GetNumberOfPoints()

    @property
    def n_cells(self) -> int:
        """Return the number of cells in the entire dataset.

        Notes
        -----
        This is identical to :attr:`n_faces <vtk_override.datamodel.pointset.poly.PolyData.n_faces>`
        in :class:`vtk.vtkPolyData`.

        Examples
        --------
        Create a mesh and return the number of cells in the
        mesh.

        >>> import vtk
        >>> from vtk_override.utils import Cube
        >>> cube = Cube()
        >>> cube.n_cells
        6

        """
        return self.GetNumberOfCells()

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return the bounding box of this dataset.

        The form is: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

        Examples
        --------
        Create a cube and return the bounds of the mesh.

        >>> import vtk
        >>> from vtk_override.utils import Cube
        >>> cube = Cube()
        >>> cube.bounds
        (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

        """
        return self.GetBounds()

    @property
    def length(self) -> float:
        """Return the length of the diagonal of the bounding box.

        Examples
        --------
        Get the length of the bounding box of a cube.  This should
        match ``3**(1/2)`` since it is the diagonal of a cube that is
        ``1 x 1 x 1``.

        >>> import vtk
        >>> from vtk_override.utils import Cube
        >>> cube = Cube()
        >>> mesh.length
        1.7320508075688772

        """
        return self.GetLength()

    @property
    def center(self) -> Vector:
        """Return the center of the bounding box.

        Examples
        --------
        Get the center of a mesh.

        >>> import vtk
        >>> from vtk_override.utils import Sphere
        >>> mesh = Sphere(center=(1, 2, 0))
        >>> mesh.center
        [1.0, 2.0, 0.0]

        """
        return list(self.GetCenter())

    def cell_points(self, ind: int) -> np.ndarray:
        """Return the points in a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        numpy.ndarray
            An array of floats with shape (number of points, 3) containing the coordinates of the
            cell corners.

        """
        # A copy of the points must be returned to avoid overlapping them since the
        # `vtk.vtkExplicitStructuredGrid.GetCell` is an override method.
        points = self.GetCell(ind).GetPoints().GetData()
        points = vtk_to_numpy(points)
        return points.copy()

    def cell_bounds(self, ind: int) -> Tuple[float, float, float, float, float, float]:
        """Return the bounding box of a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        tuple(float)
            The limits of the cell in the X, Y and Z directions respectively.

        """
        return self.GetCell(ind).GetBounds()

    def cell_type(self, ind: int) -> int:
        """Return the type of a cell.

        Parameters
        ----------
        ind : int
            Cell type ID.

        Returns
        -------
        int
            VTK cell type. See `vtkCellType.h <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_ .

        """
        return self.GetCellType(ind)

    def cell_point_ids(self, ind: int) -> List[int]:
        """Return the point ids in a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        list[int]
            Point Ids that are associated with the cell.

        """
        cell = self.GetCell(ind)
        point_ids = cell.GetPointIds()
        return [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]


@override(vtkDataSet)
class DataSet(DataSetBase, vtkDataSet):
    pass
