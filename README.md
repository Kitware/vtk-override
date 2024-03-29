# vtk-override

> *Pythonic VTK data classes*

VTK Python data class overrides to build better interoperability with NumPy
and have a more Pythonic interface to VTK. This library and its Pythonic
interface to VTK is *heavily* inspired and adopted from
[PyVista](https://github.com/pyvista/pyvista).

This is an experimental code base and is not intended for wide use and adoption.
If you are looking for a stable, Pythonic interface to VTK, please use
[PyVista](https://github.com/pyvista/pyvista).

## Limitations

This is experimental!

We worked here to extract much of PyVista's core API into a standalone package,
but found this to be no easy feat. PyVista has several thousand unit tests for
the 6,000+ lines of code in their core module so extracting all of that into
one place while keeping up with the latest changes proved to be futile.

This library copies most of PyVista's core API including the wrapping of VTK
data classes, adding dictionary-like support for point, cell, and field data,
and providng shared array access to points, cells, and data with NumPy.
Unfortunately, these interfaces are complex with tons of edge cases which are
well captured in PyVista at this time so we decided to stop this experiment
and rely on PyVista moving forward.

In time, we hope to engage with the PyVista community to streamline their
well designed core data API into a standalone package that will hopefully
become the basis for the VTK Python ecosystem.


## VTK's New Python Class Overrides

VTK 9.2 released a new "override" mechanism for VTK classes such that any
Python class that subclasses a VTK bound class can be decorated so that
all instances of the parent class generated by VTK are instances of the
subclass.

Here are some links to familiarize:

- https://discourse.vtk.org/t/more-pythonic-interface-closer-to-numpy/7905
- https://gitlab.kitware.com/berkgeveci/pythonic-vtk
- https://gitlab.kitware.com/vtk/vtk/-/merge_requests/8886
- https://gitlab.kitware.com/berkgeveci/pythonic-vtk/-/blob/master/datamodel.py
- https://gitlab.kitware.com/vtk/vtk/-/issues/18594

The interface looks like:

```py
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource


@vtkPolyData.override
class PolyData(vtkPolyData):
    """Custom Pythonic wrapping of vtkPolyData."""
    @property
    def n_points(self):
        return self.GetNumberOfPoints()
```

Instantiating a `vtkPolyData` is actually instantiated our new Python class:
```py
>>> poly = vtkPolyData()
>>> type(poly)
<class '__main__.PolyData'>
```

And any `vtkPolyData` generated by a VTK source or filter will also
be an instance of the subclass:
```py
>>> source = vtkSphereSource()
>>> source.Update()
>>> sphere = source.GetOutput()
>>> sphere.n_points
50
```


## Motivation

VTK's Python API is an automatically generated binding to C++ objects
and thus the bound Python API is nearly identical to it's C++ counterpart.
Many 3rd party libraries, namely PyVista, have done a wonderful job of
modernizing this API in Python to be more user friendly and "Pythonic".
We would like to embrace those changes into a unified, core Pythonic API
for VTK's data classes. This experimental code base is an attempt at such.

### Scope

This code base is to focus only on wrapping `vtkDataObject` subclasses
to create a user friendly API for creating and accessing VTK data objects
in Python.

Filters and plotting may come later but other libraries like PyVista and
Vedo have already done a great job on this front.



## Highlights

* Direct, shared access to points and cells as NumPy arrays
* Property access to data model attributes (e.g., `n_points`)
* Easily cast data types (e.g., `.cast_to_unstructured_grid()`)


## Usage

```py
>>> import vtk, vtk_override
>>> mesh = vtk.vtkImageData()
>>> mesh.dimensions = (2, 2, 2)
>>> mesh.n_points
8
>>> mesh.points
array([[0., 0., 0.],
       [1., 0., 0.],
       [0., 1., 0.],
       [1., 1., 0.],
       [0., 0., 1.],
       [1., 0., 1.],
       [0., 1., 1.],
       [1., 1., 1.]])
```

```py
>>> from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
>>> wavelet_source = vtkRTAnalyticSource()
>>> wavelet_source.Update()
>>> image = wavelet_source.GetOutput()
>>> image.n_points
9261
>>> image.points
array([[-10., -10., -10.],
       [ -9., -10., -10.],
       [ -8., -10., -10.],
       ...,
       [  8.,  10.,  10.],
       [  9.,  10.,  10.],
       [ 10.,  10.,  10.]])
```

## Installation

This currently requires VTK >= 9.2

```bash
pip install 'vtk>=9.2.0rc2'
```

You will need to clone this repo to install with `pip install -e .`

## Feedback and Contributing

*Do you want more from this library?* What you want might already be implemented in PyVista, so check there first.
