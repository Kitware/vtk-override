import collections.abc
import enum
from typing import Optional, Tuple, Union
import warnings

import numpy as np
from vtkmodules.vtkCommonCore import (
    vtkAbstractArray,
    vtkBitArray,
    vtkCharArray,
    vtkDataArray,
    vtkPoints,
    vtkStringArray,
)
from vtkmodules.vtkCommonDataModel import vtkDataObject

from vtk_override.utils._typing import ID_TYPE, NumericArray, VectorArray
from vtk_override.version import VTK9

if VTK9:
    from vtkmodules.util.numpy_support import (
        get_vtk_array_type,
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )
else:
    from vtk.util.numpy_support import (
        get_vtk_array_type,
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )


class FieldAssociation(enum.Enum):
    """Represents which type of vtk field a scalar or vector array is associated with."""

    POINT = vtkDataObject.FIELD_ASSOCIATION_POINTS
    CELL = vtkDataObject.FIELD_ASSOCIATION_CELLS
    NONE = vtkDataObject.FIELD_ASSOCIATION_NONE
    ROW = vtkDataObject.FIELD_ASSOCIATION_ROWS


def get_vtk_type(typ) -> int:
    """Look up the VTK type for a given numpy data type.

    Corrects for string type mapping issues.

    Parameters
    ----------
    typ : numpy.dtype
        Numpy data type.

    Returns
    -------
    int
        Integer type id specified in ``vtkType.h``

    """
    typ = get_vtk_array_type(typ)
    # This handles a silly string type bug
    if typ == 3:
        return 13
    return typ


def vtk_bit_array_to_char(vtkarr_bint) -> vtkCharArray:
    """Cast vtk bit array to a char array.

    Parameters
    ----------
    vtkarr_bint : vtk.vtkBitArray
        VTK binary array.

    Returns
    -------
    vtk.vtkCharArray
        VTK char array.

    Notes
    -----
    This performs a copy.

    """
    vtkarr = vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def vtk_id_list_to_array(vtk_id_list) -> np.ndarray:
    """Convert a vtkIdList to a NumPy array.

    Parameters
    ----------
    vtk_id_list : vtk.vtkIdList
        VTK ID list.

    Returns
    -------
    numpy.ndarray
        Array of IDs.

    """
    return np.array([vtk_id_list.GetId(i) for i in range(vtk_id_list.GetNumberOfIds())])


def convert_string_array(arr, name=None):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    Parameters
    ----------
    arr : numpy.ndarray
        Numpy string array to convert.

    name : str, optional
        Name to set the vtkStringArray to.

    Returns
    -------
    vtkStringArray
        VTK string array.

    Notes
    -----
    Note that this is terribly inefficient. If you have ideas on how
    to make this faster, please consider opening a pull request.

    """
    if isinstance(arr, np.ndarray):
        vtkarr = vtkStringArray()
        ########### OPTIMIZE ###########
        for val in arr:
            vtkarr.InsertNextValue(val)
        ################################
        if isinstance(name, str):
            vtkarr.SetName(name)
        return vtkarr
    # Otherwise it is a vtk array and needs to be converted back to numpy
    ############### OPTIMIZE ###############
    nvalues = arr.GetNumberOfValues()
    return np.array([arr.GetValue(i) for i in range(nvalues)], dtype="|U")
    ########################################


def convert_array(arr, name: str = None, deep: bool = False, array_type: Optional[int] = None):
    """Convert a NumPy array to a vtkDataArray or vice versa.

    Parameters
    ----------
    arr : np.ndarray or vtkDataArray
        A numpy array or vtkDataArry to convert.
    name : str, optional
        The name of the data array for VTK.
    deep : bool, optional
        If input is numpy array then deep copy values.
    array_type : int, optional
        VTK array type ID as specified in specified in ``vtkType.h``.

    Returns
    -------
    vtkDataArray, numpy.ndarray, or DataFrame
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns ``vtkDataArray`` or is input is ``vtkDataArray`` then
        returns NumPy ``ndarray``.

    """
    if arr is None:
        return
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.dtype("O"):
            arr = arr.astype("|S")
        arr = np.ascontiguousarray(arr)
        if arr.dtype.type in (np.str_, np.bytes_):
            # This handles strings
            vtk_data = convert_string_array(arr)
        else:
            # This will handle numerical data
            arr = np.ascontiguousarray(arr)
            vtk_data = numpy_to_vtk(num_array=arr, deep=deep, array_type=array_type)
        if isinstance(name, str):
            vtk_data.SetName(name)
        return vtk_data
    # Otherwise input must be a vtkDataArray
    if not isinstance(arr, (vtkDataArray, vtkBitArray, vtkStringArray)):
        raise TypeError(f"Invalid input array type ({type(arr)}).")
    # Handle booleans
    if isinstance(arr, vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return vtk_to_numpy(arr)


def copy_vtk_array(array, deep=True):
    """Create a deep or shallow copy of a VTK array.

    Parameters
    ----------
    array : vtk.vtkDataArray or vtk.vtkAbstractArray
        VTK array.

    deep : bool, optional
        When ``True``, create a deep copy of the array. When ``False``, returns
        a shallow copy.

    Returns
    -------
    vtk.vtkDataArray or vtk.vtkAbstractArray
        Copy of the original VTK array.

    Examples
    --------
    Perform a deep copy of a vtk array.

    >>> import vtk
    >>> from vtk_override.utils.arrays import copy_vtk_array
    >>> arr = vtk.vtkFloatArray()
    >>> _ = arr.SetNumberOfValues(10)
    >>> arr.SetValue(0, 1)
    >>> arr_copy = copy_vtk_array(arr)
    >>> arr_copy.GetValue(0)
    1.0

    """
    if not isinstance(array, (vtkDataArray, vtkAbstractArray)):
        raise TypeError(f"Invalid type {type(array)}.")

    new_array = type(array)()
    if deep:
        new_array.DeepCopy(array)
    else:
        new_array.ShallowCopy(array)

    return new_array


def vtk_points(points, deep=True, force_float=False):
    """Convert numpy array or array-like to a ``vtkPoints`` object.

    Parameters
    ----------
    points : numpy.ndarray or sequence
        Points to convert.  Should be 1 or 2 dimensional.  Accepts a
        single point or several points.

    deep : bool, optional
        Perform a deep copy of the array.  Only applicable if
        ``points`` is a :class:`numpy.ndarray`.

    force_float : bool, optional
        Casts the datatype to ``float32`` if points datatype is
        non-float.  Set this to ``False`` to allow non-float types,
        though this may lead to truncation of intermediate floats
        when transforming datasets.

    Returns
    -------
    vtk.vtkPoints
        The vtkPoints object.

    Examples
    --------
    >>> from vtk_override.utils.arrays import vtk_points
    >>> import numpy as np
    >>> points = np.random.random((10, 3))
    >>> vpoints = vtk_points(points)
    >>> vpoints  # doctest:+SKIP
    (vtkmodules.vtkCommonCore.vtkPoints)0x7f0c2e26af40

    """
    points = np.asanyarray(points)

    # verify is numeric
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError(f"Points must be a numeric type, not: {points.dtype}")

    if force_float:
        if not np.issubdtype(points.dtype, np.floating):
            warnings.warn(
                "Points is not a float type. This can cause issues when "
                "transforming or applying filters. Casting to "
                "``np.float32``. Disable this by passing "
                "``force_float=False``."
            )
            points = points.astype(np.float32)

    # check dimensionality
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    elif points.ndim > 2:
        raise ValueError(f"Dimension of ``points`` should be 1 or 2, not {points.ndim}")

    # verify shape
    if points.shape[1] != 3:
        raise ValueError(
            "Points array must contain three values per point. "
            f"Shape is {points.shape} and should be (X, 3)"
        )

    # points must be contiguous
    points = np.require(points, requirements=["C"])
    vtkpts = vtkPoints()
    vtk_arr = numpy_to_vtk(points, deep=deep)
    vtkpts.SetData(vtk_arr)
    return vtkpts


def numpy_to_idarr(ind, deep=False, return_ind=False):
    """Safely convert a numpy array to a vtkIdTypeArray."""
    ind = np.asarray(ind)

    # np.asarray will eat anything, so we have to weed out bogus inputs
    if not issubclass(ind.dtype.type, (np.bool_, np.integer)):
        raise TypeError("Indices must be either a mask or an integer array-like")

    if ind.dtype == np.bool_:
        ind = ind.nonzero()[0].astype(ID_TYPE)
    elif ind.dtype != ID_TYPE:
        ind = ind.astype(ID_TYPE)
    elif not ind.flags["C_CONTIGUOUS"]:
        ind = np.ascontiguousarray(ind, dtype=ID_TYPE)

    # must ravel or segfault when saving MultiBlock
    vtk_idarr = numpy_to_vtkIdTypeArray(ind.ravel(), deep=deep)
    if return_ind:
        return vtk_idarr, ind
    return vtk_idarr


def coerce_pointslike_arg(
    points: Union[NumericArray, VectorArray], copy: bool = False
) -> Tuple[np.ndarray, bool]:
    """Check and coerce arg to (n, 3) np.ndarray.

    Parameters
    ----------
    points : Sequence(float) or np.ndarray
        Argument to coerce into (n, 3) ``np.ndarray``.

    copy : bool, optional
        Whether to copy the ``points`` array.  Copying always occurs if ``points``
        is not ``np.ndarray``.

    Returns
    -------
    np.ndarray
        Size (n, 3) array.
    bool
        Whether the input was a single point in an array-like with shape (3,).

    """
    if isinstance(points, collections.abc.Sequence):
        points = np.asarray(points)

    if not isinstance(points, np.ndarray):
        raise TypeError("Given points must be a sequence or an array.")

    if points.ndim > 2:
        raise ValueError("Array of points must be 1D or 2D")

    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("Array of points must have three values per point (shape (n, 3))")
        singular = False

    else:
        if points.size != 3:
            raise ValueError("Given point must have three values")
        singular = True
        points = np.reshape(points, [1, 3])

    if copy:
        return points.copy(), singular
    return points, singular
