import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCenterOfMass

from vtk_override.datamodel.dataset import DataSetBase
from vtk_override.utils import override
from vtk_override.utils.arrays import vtk_points


class PointSetBase(DataSetBase):
    def center_of_mass(self, scalars_weight=False):
        """Return the coordinates for the center of mass of the mesh.

        Parameters
        ----------
        scalars_weight : bool, optional
            Flag for using the mesh scalars as weights. Defaults to ``False``.

        Returns
        -------
        numpy.ndarray
            Coordinates for the center of mass.

        Examples
        --------
        >>> import vtk
        >>> source = vtkSphereSource()
        >>> source.SetRadius(1.0)
        >>> source.SetCenter((1, 1, 1))
        >>> source.Update()
        >>> mesh = source.GetOutput()
        >>> mesh.center_of_mass()
        array([1., 1., 1.])

        """
        alg = vtkCenterOfMass()
        alg.SetInputDataObject(self)
        alg.SetUseScalarsAsWeights(scalars_weight)
        alg.Update()
        return np.array(alg.GetCenter())

    def points_to_double(self):
        """Convert the points datatype to double precision.

        Returns
        -------
        vtk.vtkPointSet
            Pointset with points in double precision.

        Notes
        -----
        This operates in place.

        Examples
        --------
        Create a mesh that has points of the type ``float32`` and
        convert the points to ``float64``.

        >>> import vtk
        >>> source = vtkSphereSource()
        >>> source.SetRadius(1.0)
        >>> source.SetCenter((1, 1, 1))
        >>> source.Update()
        >>> mesh = source.GetOutput()
        >>> mesh.points.dtype
        dtype('float32')
        >>> _ = mesh.points_to_double()
        >>> mesh.points.dtype
        dtype('float64')

        """
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)
        return self


@override(vtkPointSet)
class PointSet(PointSetBase, vtkPointSet):
    def cast_to_polydata(self, deep=True):
        """Cast this dataset to polydata.

        Parameters
        ----------
        deep : bool, optional
            Whether to copy the pointset points, or to create a PolyData
            without copying them.  Setting ``deep=True`` ensures that the
            original arrays can be modified outside the PolyData without
            affecting the PolyData. Default is ``True``.

        Returns
        -------
        vtk.vtkPolyData
            PointSet cast to a ``vtk.vtkPolyData``.

        """
        pdata = vtkPolyData()
        pdata.SetPoints(vtk_points(self.points, deep=deep, force_float=True))
        if deep:
            pdata.point_data.update(self.point_data)  # update performs deep copy
        else:
            for key, value in self.point_data.items():
                pdata.point_data[key] = value
        pdata.make_vertex_cells()
        return pdata
