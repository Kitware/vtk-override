from typing import Sequence, Tuple, Union

import numpy as np
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkRectilinearGrid,
    vtkStructuredGrid,
    vtkStructuredPoints,
    vtkUniformGrid,
)
from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid

from vtk_override.datamodel.dataset import DataSetBase
from vtk_override.utils import override


class ImageDataBase(DataSetBase):
    def __init__(self, **properties):
        for name, value in properties.items():
            self.set_property(name, value)

    def set_property(self, name, value):
        from collections.abc import Iterable

        parts = name.split("_")
        method = "Set"
        for part in [part.capitalize() for part in parts]:
            method += part
        if isinstance(value, Iterable):
            getattr(self, method)(*value)
        else:
            getattr(self, method)(value)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return the grid's dimensions.

        These are effectively the number of points along each of the
        three dataset axes.

        Examples
        --------
        Create a uniform grid with dimensions ``(1, 2, 3)``.

        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(2, 3, 4))
        >>> grid.dimensions
        (2, 3, 4)

        Set the dimensions to ``(3, 4, 5)``

        >>> grid.dimensions = (3, 4, 5)

        """
        return self.GetDimensions()

    @dimensions.setter
    def dimensions(self, dims: Sequence[int]):
        """Set the dataset dimensions."""
        self.SetDimensions(*dims)
        self.Modified()

    @property  # type: ignore
    def points(self) -> np.ndarray:  # type: ignore
        """Build a copy of the implicitly defined points as a numpy array.

        Notes
        -----
        The ``points`` for a :class:`vtk.vtkImageData` cannot be set.

        Examples
        --------
        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(2, 2, 2))
        >>> grid.points
        array([[0., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [1., 1., 0.],
               [0., 0., 1.],
               [1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 1.]])

        """
        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = np.array(self.origin) + np.array(self.extent[::2])  # type: ignore
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        return np.c_[xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")]

    @points.setter
    def points(self, points):
        """Points cannot be set.

        This setter overrides the base class's setter to ensure a user does not
        attempt to set them. See https://github.com/pyvista/pyvista/issues/713.

        """
        raise AttributeError(
            "The points cannot be set. The points of "
            "`vtkImageData` are implicitly defined by the "
            "`origin`, `spacing`, and `dimensions` of the grid."
        )

    @property
    def x(self) -> np.ndarray:
        """Return all the X points.

        Examples
        --------
        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(2, 2, 2))
        >>> grid.x
        array([0., 1., 0., 1., 0., 1., 0., 1.])

        """
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return all the Y points.

        Examples
        --------
        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(2, 2, 2))
        >>> grid.y
        array([0., 0., 1., 1., 0., 0., 1., 1.])

        """
        return self.points[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Return all the Z points.

        Examples
        --------
        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(2, 2, 2))
        >>> grid.z
        array([0., 0., 0., 0., 1., 1., 1., 1.])

        """
        return self.points[:, 2]

    @property
    def origin(self) -> Tuple[float]:
        """Return the origin of the grid (bottom southwest corner).

        Examples
        --------
        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(5, 5, 5))
        >>> grid.origin
        (0.0, 0.0, 0.0)

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, origin: Sequence[Union[float, int]]):
        """Set the origin."""
        self.SetOrigin(origin[0], origin[1], origin[2])
        self.Modified()

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Return or set the spacing for each axial direction.

        Notes
        -----
        Spacing must be non-negative. While VTK accepts negative
        spacing, this results in unexpected behavior. See:
        https://github.com/pyvista/pyvista/issues/1967

        Examples
        --------
        Create a 5 x 5 x 5 uniform grid.

        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(5, 5, 5))
        >>> grid.spacing
        (1.0, 1.0, 1.0)

        Modify the spacing to ``(1, 2, 3)``

        >>> grid.spacing = (1, 2, 3)
        >>> grid.spacing
        (1.0, 2.0, 3.0)

        """
        return self.GetSpacing()

    @spacing.setter
    def spacing(self, spacing: Sequence[Union[float, int]]):
        """Set spacing."""
        if min(spacing) < 0:
            raise ValueError(f"Spacing must be non-negative, got {spacing}")
        self.SetSpacing(*spacing)
        self.Modified()

    def cast_to_structured_grid(self) -> vtkStructuredGrid:
        """Cast this uniform grid to a structured grid.

        Returns
        -------
        vtk.vtkStructuredGrid
            This grid as a structured grid.

        """
        alg = vtkImageToStructuredGrid()
        alg.SetInputData(self)
        alg.Update()
        return alg.GetOutput()

    def cast_to_rectilinear_grid(self) -> vtkRectilinearGrid:
        """Cast this uniform grid to a rectilinear grid.

        Returns
        -------
        vtk.vtkRectilinearGrid
            This uniform grid as a rectilinear grid.

        """

        def gen_coords(i):
            coords = (
                np.cumsum(np.insert(np.full(self.dimensions[i] - 1, self.spacing[i]), 0, 0))
                + self.origin[i]
            )
            return coords

        xcoords = gen_coords(0)
        ycoords = gen_coords(1)
        zcoords = gen_coords(2)
        grid = vtkRectilinearGrid()
        grid.x = xcoords
        grid.y = ycoords
        grid.z = zcoords
        grid.point_data.update(self.point_data)
        grid.cell_data.update(self.cell_data)
        grid.field_data.update(self.field_data)
        return grid

    @property
    def extent(self) -> tuple:
        """Return or set the extent of the vtkImageData.

        The extent is simply the first and last indices for each of the three axes.

        Examples
        --------
        Create a ``vtkImageData`` and show its extent.

        >>> import vtk
        >>> grid = vtk.vtkImageData(dimensions=(10, 10, 10))
        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        >>> grid.extent = (2, 5, 2, 5, 2, 5)
        >>> grid.extent
        (2, 5, 2, 5, 2, 5)

        Note how this also modifies the grid bounds and dimensions. Since we
        use default spacing of 1 here, the bounds match the extent exactly.

        >>> grid.bounds
        (2.0, 5.0, 2.0, 5.0, 2.0, 5.0)
        >>> grid.dimensions
        (4, 4, 4)

        """
        return self.GetExtent()

    @extent.setter
    def extent(self, new_extent: Sequence[int]):
        """Set the extent of the vtkImageData."""
        if len(new_extent) != 6:
            raise ValueError("Extent must be a vector of 6 values.")
        self.SetExtent(new_extent)


@override(vtkImageData)
class ImageData(ImageDataBase, vtkImageData):
    pass


@override(vtkStructuredPoints)
class StructuredPoints(ImageDataBase, vtkStructuredPoints):
    pass


@override(vtkUniformGrid)
class UniformGrid(ImageDataBase, vtkUniformGrid):
    pass
