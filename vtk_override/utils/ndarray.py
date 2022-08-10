"""Contains vtk_ndarray a numpy ndarray type."""
from collections.abc import Iterable
from typing import Union

import numpy as np
from vtkmodules.vtkCommonCore import vtkAbstractArray, vtkWeakReference

from vtk_override.utils.arrays import FieldAssociation, convert_array
from vtk_override.version import VTK9

if VTK9:
    from vtkmodules.numpy_interface.dataset_adapter import VTKArray, VTKObjectWrapper
else:
    from vtk.numpy_interface.dataset_adapter import VTKArray, VTKObjectWrapper


class vtk_ndarray(np.ndarray):
    """An ndarray which references the owning dataset and the underlying vtkArray."""

    def __new__(
        cls,
        array: Union[Iterable, vtkAbstractArray],
        dataset=None,
        association=FieldAssociation.NONE,
    ):
        """Allocate the array."""
        if isinstance(array, Iterable):
            obj = np.asarray(array).view(cls)
        elif isinstance(array, vtkAbstractArray):
            obj = convert_array(array).view(cls)
            obj.VTKObject = array
        else:
            raise TypeError(
                f"vtk_ndarray got an invalid type {type(array)}. "
                "Expected an Iterable or vtk.vtkAbstractArray"
            )

        obj.association = association
        obj.dataset = vtkWeakReference()
        if isinstance(dataset, VTKObjectWrapper):
            obj.dataset.Set(dataset.VTKObject)
        else:
            obj.dataset.Set(dataset)
        return obj

    def __array_finalize__(self, obj):
        """Finalize array (associate with parent metadata)."""
        # this is necessary to ensure that views/slices of vtk_ndarray
        # objects stay associated with those of their parents.
        #
        # the VTKArray class uses attributes called `DataSet` and `Assocation`
        # to hold this data. I don't know why this class doesn't use the same
        # convention, but here we just map those over to the appropriate
        # attributes of this class
        VTKArray.__array_finalize__(self, obj)
        if np.shares_memory(self, obj):
            self.dataset = getattr(obj, "dataset", None)
            self.association = getattr(obj, "association", FieldAssociation.NONE)
            self.VTKObject = getattr(obj, "VTKObject", None)
        else:
            self.dataset = None
            self.association = FieldAssociation.NONE
            self.VTKObject = None

    def __setitem__(self, key: Union[int, np.ndarray], value):
        """Implement [] set operator.

        When the array is changed it triggers "Modified()" which updates
        all upstream objects, including any render windows holding the
        object.
        """
        super().__setitem__(key, value)
        if self.VTKObject is not None:
            self.VTKObject.Modified()

        # the associated dataset should also be marked as modified
        dataset = self.dataset
        if dataset is not None and dataset.Get():
            dataset.Get().Modified()

    def __array_wrap__(self, out_arr, context=None):
        """Return a numpy scalar if array is 0d.

        See https://github.com/numpy/numpy/issues/5819

        """
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)

        # Match numpy's behavior and return a numpy dtype scalar
        return out_arr[()]

    __getattr__ = VTKArray.__getattr__
