from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
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
