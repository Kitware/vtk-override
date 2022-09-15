from pathlib import Path

from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkPlaneSource, vtkSphereSource
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource


def Sphere(radius=0.5, center=(0, 0, 0)):
    sphere = vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetCenter(center)
    sphere.SetThetaResolution(30)
    sphere.SetPhiResolution(30)
    sphere.SetStartTheta(0)
    sphere.SetEndTheta(260)
    sphere.SetStartPhi(0)
    sphere.SetEndPhi(180)
    sphere.Update()
    return sphere.GetOutput()


def Cube(center=(0.0, 0.0, 0.0)):
    src = vtkCubeSource()
    src.SetCenter(center)
    src.SetXLength(1.0)
    src.SetYLength(1.0)
    src.SetZLength(1.0)
    src.Update()
    return src.GetOutput()


def Wavelet():
    wavelet_source = vtkRTAnalyticSource()
    wavelet_source.Update()
    return wavelet_source.GetOutput()


def Plane(origin=(0, 0, 0), normal=(0, 0, 1), i_size=1, j_size=1, i_resolution=10, j_resolution=10):
    """Create a plane.

    Parameters
    ----------
    origin : list or tuple or np.ndarray
        Location of the centroid in ``[x, y, z]``.

    normal : list or tuple or np.ndarray
        Direction of the plane's normal in ``[x, y, z]``.

    i_size : float
        Size of the plane in the i direction.

    j_size : float
        Size of the plane in the j direction.

    i_resolution : int
        Number of points on the plane in the i direction.

    j_resolution : int
        Number of points on the plane in the j direction.

    Returns
    -------
    vtk.vtkPolyData
        Plane mesh.

    Examples
    --------
    Create a default plane.

    >>> from vtk_override.utils.sources import Plane
    >>> mesh = Plane()
    >>> mesh.point_data.clear()
    >>> mesh.plot(show_edges=True)
    """
    planeSource = vtkPlaneSource()
    planeSource.SetOrigin(origin)
    planeSource.SetNormal(normal)
    planeSource.SetXResolution(i_resolution)
    planeSource.SetYResolution(j_resolution)
    planeSource.Update()

    surf = planeSource.GetOutput()

    surf.points[:, 0] *= i_size
    surf.points[:, 1] *= j_size
    return surf


def Hexbeam():
    path = str(Path(__file__).parent / "hexbeam.vtk")
    reader = vtkDataSetReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()
