import numpy as np
import pytest

from vtk_override.datamodel.image import ImageData
from vtk_override.datamodel.pointset.structured import StructuredGrid
from vtk_override.datamodel.rectilinear import RectilinearGrid


def test_image_data_override(wavelet):
    assert isinstance(wavelet, ImageData)
    assert isinstance(wavelet.dimensions, tuple)
    assert isinstance(wavelet.origin, tuple)
    assert isinstance(wavelet.spacing, tuple)
    assert isinstance(wavelet.extent, tuple)


def test_image_data_points():
    image = ImageData()
    image.dimensions = (2, 2, 2)
    assert isinstance(image.points, np.ndarray)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(image.points, points)
    with pytest.raises(AttributeError):
        image.points = np.random.rand(image.n_points, 3)
    assert np.allclose(image.x, points[:, 0])
    assert np.allclose(image.y, points[:, 1])
    assert np.allclose(image.z, points[:, 2])


def test_image_data_properties(wavelet):
    wavelet.origin = (1, 2, 3)
    assert wavelet.GetOrigin() == (1, 2, 3)
    wavelet.spacing = (1, 5, 10)
    assert wavelet.GetSpacing() == (1, 5, 10)
    assert wavelet.extent
    wavelet.extent = wavelet.extent


def test_cast_image_to_structured_grid():
    image = ImageData()
    image.dimensions = (2, 2, 2)
    grid = image.cast_to_structured_grid()
    assert isinstance(grid, StructuredGrid)
    assert grid.dimensions == image.dimensions
    assert isinstance(grid.points, np.ndarray)


def test_cast_image_to_rectilinear_grid():
    image = ImageData()
    image.dimensions = (2, 2, 2)
    image.spacing = (1, 2, 3)
    grid = image.cast_to_rectilinear_grid()
    assert isinstance(grid, RectilinearGrid)
    assert grid.dimensions == image.dimensions
    assert isinstance(grid.points, np.ndarray)
    assert np.allclose(grid.x, [0, 1])
    assert np.allclose(grid.y, [0, 2])
    assert np.allclose(grid.z, [0, 3])


def test_image_eq(wavelet):
    copy = wavelet.copy(deep=True)
    copy.origin = [1, 1, 1]
    assert wavelet != copy

    copy.origin = [0, 0, 0]
    assert wavelet == copy

    copy.point_data.clear()
    assert wavelet != copy
