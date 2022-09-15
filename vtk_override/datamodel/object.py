import collections
from typing import DefaultDict

from vtkmodules.vtkCommonDataModel import vtkDataObject

from vtk_override.datamodel.datasetattributes import DataSetAttributes
from vtk_override.utils import override
from vtk_override.utils.arrays import FieldAssociation


class DataObjectBase:
    """A wrapper for vtkDataObject that makes it easier to access FieldData
    arrays as VTKArrays
    """

    @property
    def _association_bitarray_names(self) -> DefaultDict:
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        if not hasattr(self, "__association_bitarray_names"):
            self.__association_bitarray_names = collections.defaultdict(set)
        # TODO: copy these attributes in `copy` methods
        return self.__association_bitarray_names

    @property
    def _association_complex_names(self) -> DefaultDict:
        # view these arrays as complex128 as VTK doesn't support complex types
        if not hasattr(self, "__association_complex_names"):
            self.__association_complex_names = collections.defaultdict(set)
        # TODO: copy these attributes in `copy` methods
        return self.__association_complex_names

    def shallow_copy(self, to_copy: vtkDataObject) -> vtkDataObject:
        """Shallow copy the given mesh to this mesh.

        Parameters
        ----------
        to_copy : vtk.vtkDataObject
            Data object to perform a shallow copy from.

        """
        self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy: vtkDataObject) -> vtkDataObject:
        """Overwrite this data object with another data object as a deep copy.

        Parameters
        ----------
        to_copy : vtk.vtkDataObject
            Data object to perform a deep copy from.

        """
        if not isinstance(self, type(to_copy)):
            raise TypeError(
                f"The Input DataSet type {type(to_copy)} must be "
                f"compatible with the one being overwritten {type(self)}"
            )
        self.DeepCopy(to_copy)

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            When ``True`` makes a full copy of the object.  When
            ``False``, performs a shallow copy where the points, cell,
            and data arrays are references to the original object.

        Returns
        -------
        vtkDataObject
            Deep or shallow copy of the input.  Type is identical to
            the input.

        Examples
        --------
        Create and make a deep copy of a PolyData object.

        >>> import vtk
        >>> mesh_a = vtk.vtkPolyData()
        >>> mesh_b = mesh_a.copy()
        >>> mesh_a == mesh_b
        True

        """
        thistype = type(self)
        newobject = thistype()

        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        # newobject.copy_meta_from(self, deep)
        return newobject

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def memory_address(self) -> str:
        """Get address of the underlying VTK C++ object.

        Returns
        -------
        str
            Memory address formatted as ``'Addr=%p'``.

        Examples
        --------
        >>> import vtk
        >>> mesh = vtk.vtkPolyData()
        >>> mesh.memory_address
        'Addr=...'

        """
        return self.GetInformation().GetAddressAsString("")

    @property
    def actual_memory_size(self) -> int:
        """Return the actual size of the dataset object.

        Returns
        -------
        int
            The actual size of the dataset object in kibibytes (1024
            bytes).

        Examples
        --------
        >>> import vtk
        >>> mesh = vtk.vtkPolyData()
        >>> mesh.actual_memory_size  # doctest:+SKIP
        93

        """
        return self.GetActualMemorySize()

    @property
    def field_data(self) -> DataSetAttributes:
        """Return FieldData as DataSetAttributes.

        Use field data when size of the data you wish to associate
        with the dataset does not match the number of points or cells
        of the dataset.

        Examples
        --------
        Add field data to a PolyData dataset and then return it.

        >>> from vtk_override.utils.sources import Sphere
        >>> import numpy as np
        >>> mesh = Sphere()
        >>> mesh.field_data['my-field-data'] = np.arange(10)
        >>> mesh.field_data['my-field-data']
        vtk_ndarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        """
        return DataSetAttributes(
            self.GetFieldData(), dataset=self, association=FieldAssociation.NONE
        )

    def clear_field_data(self):
        """Remove all field data.

        Examples
        --------
        Add field data to a PolyData dataset and then remove it.

        >>> from vtk_override.utils.sources import Sphere
        >>> mesh = Sphere()
        >>> mesh.field_data['my-field-data'] = range(10)
        >>> len(mesh.field_data)
        1
        >>> mesh.clear_field_data()
        >>> len(mesh.field_data)
        0

        """
        self.field_data.clear()


@override(vtkDataObject)
class DataObject(DataObjectBase, vtkDataObject):
    pass
