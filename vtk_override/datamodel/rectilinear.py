from typing import Sequence, Tuple

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid, vtkStructuredGrid
from vtkmodules.vtkFiltersGeneral import vtkRectilinearGridToPointSet

from vtk_override.datamodel.dataset import DataSetBase
from vtk_override.utils import convert_array, override


@override(vtkRectilinearGrid)
class RectilinearGrid(DataSetBase, vtkRectilinearGrid):
    def __init__(self, x=None, y=None, z=None):
        super().__init__()
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if (x, y, z).count(None) < 3:
            self._update_dimensions()

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return the grid's dimensions.

        These are effectively the number of points along each of the
        three dataset axes.

        Examples
        --------
        Create a uniform grid with dimensions ``(1, 2, 3)``.

        >>> import vtk
        >>> grid = vtk.vtkImageData()
        >>> grid.dimensions = (2, 3, 4)
        >>> grid.dimensions
        (2, 3, 4)

        Set the dimensions to ``(3, 4, 5)``

        >>> grid.dimensions = (3, 4, 5)

        """
        return self.GetDimensions()

    @dimensions.setter  # type: ignore
    def dimensions(self, dims):
        """Dimensions of the RectilinearGrid cannot be set."""
        raise AttributeError(
            "The dimensions of a `RectilinearGrid` are implicitly "
            "defined and thus cannot be set."
        )

    def _update_dimensions(self):
        """Update the dimensions if coordinates have changed."""
        return self.SetDimensions(len(self.x), len(self.y), len(self.z))

    @property
    def meshgrid(self) -> list:
        """Return a meshgrid of numpy arrays for this mesh.

        This simply returns a :func:`numpy.meshgrid` of the
        coordinates for this mesh in ``ij`` indexing. These are a copy
        of the points of this mesh.

        """
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    @property  # type: ignore
    def points(self) -> np.ndarray:  # type: ignore
        """Return a copy of the points as an n by 3 numpy array.

        Notes
        -----
        Points of a :class:`vtk.vtkRectilinearGrid` cannot be
        set. Set point coordinates with :attr:`RectilinearGrid.x`,
        :attr:`RectilinearGrid.y`, or :attr:`RectilinearGrid.z`.

        Examples
        --------
        >>> import numpy as np
        >>> import vtk
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = vtk.vtkRectilinearGrid(xrng, yrng, zrng)
        >>> grid.points
        array([[-10., -10., -10.],
               [  0., -10., -10.],
               [-10.,   0., -10.],
               [  0.,   0., -10.],
               [-10., -10.,   0.],
               [  0., -10.,   0.],
               [-10.,   0.,   0.],
               [  0.,   0.,   0.]])

        """
        xx, yy, zz = self.meshgrid
        return np.c_[xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")]

    @points.setter
    def points(self, points):
        """Raise an AttributeError.

        This setter overrides the base class's setter to ensure a user
        does not attempt to set them.
        """
        raise AttributeError(
            "The points cannot be set. The points of "
            "`RectilinearGrid` are defined in each axial direction. Please "
            "use the `x`, `y`, and `z` setters individually."
        )

    @property
    def x(self) -> np.ndarray:
        """Return or set the coordinates along the X-direction.

        Examples
        --------
        Return the x coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import vtk
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = vtk.vtkRectilinearGrid(xrng, yrng, zrng)
        >>> grid.x
        array([-10.,   0.])

        Set the x coordinates of a RectilinearGrid.

        >>> grid.x = [-10.0, 0.0, 10.0]
        >>> grid.x
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetXCoordinates())

    @x.setter
    def x(self, coords: Sequence):
        """Set the coordinates along the X-direction."""
        self.SetXCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def y(self) -> np.ndarray:
        """Return or set the coordinates along the Y-direction.

        Examples
        --------
        Return the y coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import vtk
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = vtk.vtkRectilinearGrid(xrng, yrng, zrng)
        >>> grid.y
        array([-10.,   0.])

        Set the y coordinates of a RectilinearGrid.

        >>> grid.y = [-10.0, 0.0, 10.0]
        >>> grid.y
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetYCoordinates())

    @y.setter
    def y(self, coords: Sequence):
        """Set the coordinates along the Y-direction."""
        self.SetYCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def z(self) -> np.ndarray:
        """Return or set the coordinates along the Z-direction.

        Examples
        --------
        Return the z coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import vtk
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = vtk.vtkRectilinearGrid(xrng, yrng, zrng)
        >>> grid.z
        array([-10.,   0.])

        Set the z coordinates of a RectilinearGrid.

        >>> grid.z = [-10.0, 0.0, 10.0]
        >>> grid.z
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetZCoordinates())

    @z.setter
    def z(self, coords: Sequence):
        """Set the coordinates along the Z-direction."""
        self.SetZCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    def cast_to_structured_grid(self) -> vtkStructuredGrid:
        """Cast this rectilinear grid to a structured grid.

        Returns
        -------
        vtk.vtkStructuredGrid
            This grid as a structured grid.

        """
        alg = vtkRectilinearGridToPointSet()
        alg.SetInputData(self)
        alg.Update()
        return alg.GetOutput()
