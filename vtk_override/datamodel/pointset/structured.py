from vtkmodules.vtkCommonDataModel import vtkStructuredGrid

from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override


@override(vtkStructuredGrid)
class StructuredGrid(PointSetBase, vtkStructuredGrid):
    @property
    def dimensions(self):
        """Return a length 3 tuple of the grid's dimensions.

        Returns
        -------
        tuple
            Grid dimensions.

        Examples
        --------
        >>> import vtk
        >>> import numpy as np
        >>> xrng = np.arange(-10, 10, 1, dtype=np.float32)
        >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
        >>> zrng = np.arange(-10, 10, 5, dtype=np.float32)
        >>> x, y, z = np.meshgrid(xrng, yrng, zrng)
        >>> grid = vtk.vtkStructuredGrid(x, y, z)
        >>> grid.dimensions
        (10, 20, 4)

        """
        return tuple(self.GetDimensions())

    @dimensions.setter
    def dimensions(self, dims):
        """Set the dataset dimensions. Pass a length three tuple of integers."""
        nx, ny, nz = dims[0], dims[1], dims[2]
        self.SetDimensions(nx, ny, nz)
        self.Modified()
