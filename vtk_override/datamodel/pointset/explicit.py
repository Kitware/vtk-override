from typing import Tuple

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid, vtkUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkExplicitStructuredGridToUnstructuredGrid

from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override


@override(vtkExplicitStructuredGrid)
class ExplicitStructuredGrid(PointSetBase, vtkExplicitStructuredGrid):
    def cast_to_unstructured_grid(self) -> vtkUnstructuredGrid:
        """Cast to an unstructured grid.

        Returns
        -------
        UnstructuredGrid
            An unstructured grid. VTK adds the ``'BLOCK_I'``,
            ``'BLOCK_J'`` and ``'BLOCK_K'`` cell arrays. These arrays
            are required to restore the explicit structured grid.

        Notes
        -----
        The ghost cell array is disabled before casting the
        unstructured grid in order to allow the original structure
        and attributes data of the explicit structured grid to be
        restored. If you don't need to restore the explicit
        structured grid later or want to extract an unstructured
        grid from the visible subgrid, use the ``extract_cells``
        filter and the cell indices where the ghost cell array is
        ``0``.

        """
        grid = ExplicitStructuredGrid()
        grid.copy_structure(self)
        alg = vtkExplicitStructuredGridToUnstructuredGrid()
        alg.SetInputDataObject(grid)
        alg.Update()
        grid = alg.GetOutput()
        grid.cell_data.remove("vtkOriginalCellIds")  # unrequired
        grid.copy_attributes(self)  # copy ghost cell array and other arrays
        return grid

    def _dimensions(self):
        # This method is required to avoid conflict if a developer extends `ExplicitStructuredGrid`
        # and reimplements `dimensions` to return, for example, the number of cells in the I, J and
        # K directions.
        dims = self.GetExtent()
        dims = np.reshape(dims, (3, 2))
        dims = np.diff(dims, axis=1)
        dims = dims.flatten() + 1
        return int(dims[0]), int(dims[1]), int(dims[2])

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return the topological dimensions of the grid.

        Returns
        -------
        tuple(int)
            Number of sampling points in the I, J and Z directions respectively.

        """
        return self._dimensions()
