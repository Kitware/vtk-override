"""Type aliases for type hints."""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from vtkmodules.vtkCommonCore import vtkIdTypeArray

Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
VectorArray = Union[np.ndarray, Sequence[Vector]]
Number = Union[float, int, np.number]
NumericArray = Union[Sequence[Number], np.ndarray]


def _get_vtk_id_type():
    """Return the numpy datatype responding to ``vtk.vtkIdTypeArray``."""
    VTK_ID_TYPE_SIZE = vtkIdTypeArray().GetDataTypeSize()
    if VTK_ID_TYPE_SIZE == 4:
        return np.int32
    elif VTK_ID_TYPE_SIZE == 8:
        return np.int64
    return np.int32


ID_TYPE = _get_vtk_id_type()
