from collections import deque
from itertools import count, islice
import sys

from vtkmodules.vtkCommonDataModel import vtkCellArray

from vtk_override.utils import override
from vtk_override.utils.arrays import numpy_to_idarr, vtk_to_numpy


def ncells_from_cells_py36(cells):  # pragma: no cover
    """Get the number of cells from a VTK cell connectivity array.

    Works on all Python>=3.5
    """
    c = 0
    n_cells = 0
    while c < cells.size:
        c += cells[c] + 1
        n_cells += 1
    return n_cells


def ncells_from_cells(cells):
    """Get the number of cells from a VTK cell connectivity array.

    Works on Python>=3.7
    """
    consumer = deque(maxlen=0)
    it = cells.flat
    for n_cells in count():  # noqa: B007
        skip = next(it, None)
        if skip is None:
            break
        consumer.extend(islice(it, skip))
    return n_cells


@override(vtkCellArray)
class CellArray(vtkCellArray):
    """Wrapping of vtkCellArray.

    Provides convenience functions to simplify creating a CellArray from
    a numpy array or list.

    Import an array of data with the legacy vtkCellArray layout, e.g.

    ``{ n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ... }``
    Where n0 is the number of points in cell 0, and pX_Y is the Y'th
    point in cell X.

    Examples
    --------
    Create a cell array containing two triangles.

    >>> from vtk_override.datamodel.cells import CellArray
    >>> cellarr = CellArray([3, 0, 1, 2, 3, 3, 4, 5])
    """

    def __init__(self, cells=None, n_cells=None, deep=False):
        """Initialize a vtkCellArray."""
        if cells is not None:
            self._set_cells(cells, n_cells, deep)

    def _set_cells(self, cells, n_cells, deep):
        vtk_idarr, cells = numpy_to_idarr(cells, deep=deep, return_ind=True)

        # Get number of cells if None.  This is quite a performance
        # bottleneck and we can consider adding a warning.  Good
        # candidate for Cython or JIT compilation
        if n_cells is None:
            if cells.ndim == 1:
                if sys.version_info.minor > 6:
                    n_cells = ncells_from_cells(cells)
                else:  # pragma: no cover
                    # About 20% slower
                    n_cells = ncells_from_cells_py36(cells)
            else:
                n_cells = cells.shape[0]

        self.SetCells(n_cells, vtk_idarr)

    @property
    def cells(self):
        """Return a numpy array of the cells."""
        return vtk_to_numpy(self.GetData()).ravel()

    @property
    def n_cells(self):
        """Return the number of cells."""
        return self.GetNumberOfCells()
