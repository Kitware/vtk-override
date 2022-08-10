import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPolyData

from vtk_override.datamodel.cells import CellArray
from vtk_override.datamodel.pointset.points import PointSetBase
from vtk_override.utils import override
from vtk_override.utils.arrays import vtk_to_numpy


@override(vtkPolyData)
class PolyData(PointSetBase, vtkPolyData):
    @property
    def verts(self) -> np.ndarray:
        """Get the vertex cells.

        Returns
        -------
        numpy.ndarray
            Array of vertex cell indices.

        Examples
        --------
        Create a point cloud polydata and return the vertex cells.

        >>> import vtk
        >>> import numpy as np
        >>> points = np.random.random((5, 3))
        >>> pdata = vtk.vtkPolyData(points)
        >>> pdata.verts
        array([1, 0, 1, 1, 1, 2, 1, 3, 1, 4])

        """
        return vtk_to_numpy(self.GetVerts().GetData())

    @verts.setter
    def verts(self, verts):
        """Set the vertex cells."""
        if isinstance(verts, CellArray):
            self.SetVerts(verts)
        else:
            self.SetVerts(CellArray(verts))

    @property
    def lines(self) -> np.ndarray:
        """Return a pointer to the lines as a numpy array.

        Examples
        --------
        Return the lines from a spline.

        >>> import pyvista
        >>> import numpy as np
        >>> points = np.random.random((3, 3))
        >>> spline = pyvista.Spline(points, 10)
        >>> spline.lines
        array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

        """
        return vtk_to_numpy(self.GetLines().GetData()).ravel()

    @lines.setter
    def lines(self, lines):
        """Set the lines of the polydata."""
        if isinstance(lines, CellArray):
            self.SetLines(lines)
        else:
            self.SetLines(CellArray(lines))

    @property
    def faces(self) -> np.ndarray:
        """Return a pointer to the faces as a numpy array.

        Returns
        -------
        numpy.ndarray
            Array of face indices.

        Examples
        --------
        >>> import pyvista as pv
        >>> plane = pv.Plane(i_resolution=2, j_resolution=2)
        >>> plane.faces
        array([4, 0, 1, 4, 3, 4, 1, 2, 5, 4, 4, 3, 4, 7, 6, 4, 4, 5, 8, 7])

        Note how the faces contain a "padding" indicating the number
        of points per face:

        >>> plane.faces.reshape(-1, 5)
        array([[4, 0, 1, 4, 3],
               [4, 1, 2, 5, 4],
               [4, 3, 4, 7, 6],
               [4, 4, 5, 8, 7]])
        """
        return vtk_to_numpy(self.GetPolys().GetData())

    @faces.setter
    def faces(self, faces):
        """Set the face cells."""
        if isinstance(faces, CellArray):
            self.SetPolys(faces)
        else:
            # TODO: faster to mutate in-place if array is same size?
            self.SetPolys(CellArray(faces))

    @property
    def n_lines(self) -> int:
        """Return the number of lines.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Line()
        >>> mesh.n_lines
        1

        """
        return self.GetNumberOfLines()

    @property
    def n_verts(self) -> int:
        """Return the number of vertices.

        Examples
        --------
        Create a simple mesh containing just two points and return the
        number of vertices.

        >>> import vtk
        >>> mesh = vtk.vtkPolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> mesh.n_verts
        2

        """
        return self.GetNumberOfVerts()

    @property
    def n_faces(self) -> int:
        """Return the number of cells.

        Alias for ``n_cells``.

        Examples
        --------
        >>> import pyvista
        >>> plane = pyvista.Plane(i_resolution=2, j_resolution=2)
        >>> plane.n_faces
        4

        """
        return self.n_cells
