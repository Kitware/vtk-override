import numpy as np
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkUnstructuredGridBase

from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override
from vtk_override.utils.arrays import vtk_to_numpy


class UnstructuredGridBaseBase(PointSetBase):
    @property
    def celltypes(self) -> np.ndarray:
        """Return the cell types array.

        Returns
        -------
        numpy.ndarray
            Array of cell types.  Some of the most popular cell types:

        * ``EMPTY_CELL = 0``
        * ``VERTEX = 1``
        * ``POLY_VERTEX = 2``
        * ``LINE = 3``
        * ``POLY_LINE = 4``
        * ``TRIANGLE = 5``
        * ``TRIANGLE_STRIP = 6``
        * ``POLYGON = 7``
        * ``PIXEL = 8``
        * ``QUAD = 9``
        * ``TETRA = 10``
        * ``VOXEL = 11``
        * ``HEXAHEDRON = 12``
        * ``WEDGE = 13``
        * ``PYRAMID = 14``
        * ``PENTAGONAL_PRISM = 15``
        * ``HEXAGONAL_PRISM = 16``
        * ``QUADRATIC_EDGE = 21``
        * ``QUADRATIC_TRIANGLE = 22``
        * ``QUADRATIC_QUAD = 23``
        * ``QUADRATIC_POLYGON = 36``
        * ``QUADRATIC_TETRA = 24``
        * ``QUADRATIC_HEXAHEDRON = 25``
        * ``QUADRATIC_WEDGE = 26``
        * ``QUADRATIC_PYRAMID = 27``
        * ``BIQUADRATIC_QUAD = 28``
        * ``TRIQUADRATIC_HEXAHEDRON = 29``
        * ``QUADRATIC_LINEAR_QUAD = 30``
        * ``QUADRATIC_LINEAR_WEDGE = 31``
        * ``BIQUADRATIC_QUADRATIC_WEDGE = 32``
        * ``BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33``
        * ``BIQUADRATIC_TRIANGLE = 34``

        See
        https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
        for all cell types.

        Examples
        --------
        This mesh contains only linear hexahedral cells, type
        ``CellType.HEXAHEDRON``, which evaluates to 12.

        >>> from vtk_override.utils.sources import Hexbeam
        >>> hex_beam = Hexbeam()
        >>> hex_beam.celltypes  # doctest:+SKIP
        array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
               dtype=uint8)

        """
        return vtk_to_numpy(self.GetCellTypesArray())

    @property
    def cell_connectivity(self) -> np.ndarray:
        """Return a the vtk cell connectivity as a numpy array.

        This is effecively :attr:`UnstructuredGrid.cells` without the
        padding.

        .. note::
           This is only available in ``vtk>=9.0.0``.

        Returns
        -------
        numpy.ndarray
            Connectivity array.

        Examples
        --------
        Return the cell connectivity for the first two cells.

        >>> from vtk_override.utils.sources import Hexbeam
        >>> hex_beam = Hexbeam()
        >>> hex_beam.cell_connectivity[:16]
        array([ 0,  2,  8,  7, 27, 36, 90, 81,  2,  1,  4,  8, 36, 18, 54, 90])

        """
        return vtk_to_numpy(self.GetCells().GetConnectivityArray())

    @property
    def offset(self) -> np.ndarray:
        """Return the cell locations array.

        In VTK 9, this is the location of the start of each cell in
        :attr:`cell_connectivity`, and in VTK < 9, this is the
        location of the start of each cell in :attr:`cells`.

        Returns
        -------
        numpy.ndarray
            Array of cell offsets indicating the start of each cell.

        Examples
        --------
        Return the cell offset array within ``vtk==9``.  Since this
        mesh is composed of all hexahedral cells, note how each cell
        starts at 8 greater than the prior cell.

        >>> from vtk_override.utils.sources import Hexbeam
        >>> hex_beam = Hexbeam()
        >>> hex_beam.offset
        array([  0,   8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
               104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
               208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304,
               312, 320])

        """
        carr = self.GetCells()
        # This will be the number of cells + 1.
        return vtk_to_numpy(carr.GetOffsetsArray())


@override(vtkUnstructuredGridBase)
class UnstructuredGridBase(UnstructuredGridBaseBase, vtkUnstructuredGridBase):
    pass


@override(vtkUnstructuredGrid)
class UnstructuredGrid(UnstructuredGridBaseBase, vtkUnstructuredGrid):
    pass


# @override(vtkMappedUnstructuredGrid)
# class MappedUnstructuredGrid(UnstructuredGridBaseBase, vtkMappedUnstructuredGrid):
#     pass
