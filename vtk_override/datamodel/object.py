from vtkmodules.vtkCommonDataModel import vtkDataObject

from vtk_override.utils import override


class DataObjectBase:
    """A wrapper for vtkDataObject that makes it easier to access FieldData
    arrays as VTKArrays
    """

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
    def field_data(self):
        return super().GetFieldData()

    @field_data.setter
    def field_data(self, fd):
        return super().SetFieldData(fd)


@override(vtkDataObject)
class DataObject(DataObjectBase, vtkDataObject):
    pass
