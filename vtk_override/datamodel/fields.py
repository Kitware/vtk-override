import weakref

import numpy
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import (
    vtkCellData,
    vtkDataObject,
    vtkDataSetAttributes,
    vtkFieldData,
    vtkPointData,
)


class FieldDataBase(object):
    def __init__(self):
        self.association = None
        self.dataset = None

    def __getitem__(self, idx):
        """Implements the [] operator. Accepts an array name or index."""
        return self.get_array(idx)

    def __setitem__(self, name, value):
        """Implements the [] operator. Accepts an array name or index."""
        return self.set_array(name, value)

    def get_array(self, idx):
        "Given an index or name, returns a VTKArray."
        if isinstance(idx, int) and idx >= self.GetNumberOfArrays():
            raise IndexError("array index out of range")
        vtkarray = super().GetArray(idx)
        if not vtkarray:
            vtkarray = self.GetAbstractArray(idx)
            if vtkarray:
                return vtkarray
            return dsa.NoneArray
        array = dsa.vtkDataArrayToVTKArray(vtkarray, self.dataset)
        array.Association = self.association
        return array

    def keys(self):
        """Returns the names of the arrays as a list."""
        kys = []
        narrays = self.GetNumberOfArrays()
        for i in range(narrays):
            name = self.GetAbstractArray(i).GetName()
            if name:
                kys.append(name)
        return kys

    def values(self):
        """Returns the arrays as a list."""
        vals = []
        narrays = self.GetNumberOfArrays()
        for i in range(narrays):
            a = self.get_array(i)
            if a.GetName():
                vals.append(a)
        return vals

    def set_array(self, name, narray):
        """Appends a new array to the dataset attributes."""
        if narray is dsa.NoneArray:
            # if NoneArray, nothing to do.
            return

        if self.association == vtkDataObject.POINT:
            arrLength = self.dataset.GetNumberOfPoints()
        elif self.association == vtkDataObject.CELL:
            arrLength = self.dataset.GetNumberOfCells()
        elif self.association == vtkDataObject.ROW and self.dataset.GetNumberOfColumns() > 0:
            arrLength = self.dataset.GetNumberOfRows()
        else:
            if not isinstance(narray, numpy.ndarray):
                arrLength = 1
            else:
                arrLength = narray.shape[0]

        # Fixup input array length:
        if not isinstance(narray, numpy.ndarray) or numpy.ndim(narray) == 0:  # Scalar input
            dtype = narray.dtype if isinstance(narray, numpy.ndarray) else type(narray)
            tmparray = numpy.empty(arrLength, dtype=dtype)
            tmparray.fill(narray)
            narray = tmparray
        elif narray.shape[0] != arrLength:  # Vector input
            components = 1
            for l in narray.shape:
                components *= l
            tmparray = numpy.empty((arrLength, components), dtype=narray.dtype)
            tmparray[:] = narray.flatten()
            narray = tmparray

        shape = narray.shape

        if len(shape) == 3:
            # Array of matrices. We need to make sure the order  in memory is right.
            # If column order (c order), transpose. VTK wants row order (fortran
            # order). The deep copy later will make sure that the array is contiguous.
            # If row order but not contiguous, transpose so that the deep copy below
            # does not happen.
            size = narray.dtype.itemsize
            if (narray.strides[1] / size == 3 and narray.strides[2] / size == 1) or (
                narray.strides[1] / size == 1
                and narray.strides[2] / size == 3
                and not narray.flags.contiguous
            ):
                narray = narray.transpose(0, 2, 1)

        # If array is not contiguous, make a deep copy that is contiguous
        if not narray.flags.contiguous:
            narray = numpy.ascontiguousarray(narray)

        # Flatten array of matrices to array of vectors
        if len(shape) == 3:
            narray = narray.reshape(shape[0], shape[1] * shape[2])

        # this handle the case when an input array is directly appended on the
        # output. We want to make sure that the array added to the output is not
        # referring to the input dataset.
        copy = dsa.VTKArray(narray)
        try:
            copy.VTKObject = narray.VTKObject
        except AttributeError:
            pass
        arr = dsa.numpyTovtkDataArray(copy, name)
        self.AddArray(arr)


@vtkFieldData.override
class FieldData(FieldDataBase, vtkFieldData):
    pass


class DataSetAttributesBase(FieldDataBase):
    pass


@vtkDataSetAttributes.override
class DataSetAttributes(DataSetAttributesBase, vtkDataSetAttributes):
    pass


@vtkPointData.override
class PointData(DataSetAttributesBase, vtkPointData):
    pass


@vtkCellData.override
class CellData(DataSetAttributesBase, vtkCellData):
    pass


class CompositeDataSetAttributes(object):
    """This is a python friendly wrapper for vtkDataSetAttributes for composite
    datsets. Since composite datasets themselves don't have attribute data, but
    the attribute data is associated with the leaf nodes in the composite
    dataset, this class simulates a DataSetAttributes interface by taking a
    union of DataSetAttributes associated with all leaf nodes."""

    def __init__(self, dataset, association):
        self.DataSet = dataset
        self.Association = association
        self.ArrayNames = []
        self.Arrays = {}

        # build the set of arrays available in the composite dataset. Since
        # composite datasets can have partial arrays, we need to iterate over
        # all non-null blocks in the dataset.
        self.__determine_arraynames()

    def __determine_arraynames(self):
        array_set = set()
        array_list = []
        for dataset in self.DataSet:
            dsa = dataset.GetAttributes(self.Association)
            for array_name in dsa.keys():
                if array_name not in array_set:
                    array_set.add(array_name)
                    array_list.append(array_name)
        self.ArrayNames = array_list

    def keys(self):
        """Returns the names of the arrays as a list."""
        return self.ArrayNames

    def __getitem__(self, idx):
        """Implements the [] operator. Accepts an array name."""
        return self.get_array(idx)

    def __setitem__(self, name, narray):
        """Implements the [] operator. Accepts an array name."""
        return self.set_array(name, narray)

    def set_array(self, name, narray):
        """Appends a new array to the composite dataset attributes."""
        if narray is dsa.NoneArray:
            # if NoneArray, nothing to do.
            return

        added = False
        if not isinstance(narray, dsa.VTKCompositeDataArray):  # Scalar input
            for ds in self.DataSet:
                ds.GetAttributes(self.Association).append(narray, name)
                added = True
            if added:
                self.ArrayNames.append(name)
                # don't add the narray since it's a scalar. GetArray() will create a
                # VTKCompositeArray on-demand.
        else:
            for ds, array in zip(self.DataSet, narray.Arrays):
                if array is not None:
                    ds.GetAttributes(self.Association).set_array(name, array)
                    added = True
            if added:
                self.ArrayNames.append(name)
                self.Arrays[name] = weakref.ref(narray)

    def get_array(self, idx):
        """Given a name, returns a VTKCompositeArray."""
        arrayname = idx
        if arrayname not in self.ArrayNames:
            return dsa.NoneArray
        if arrayname not in self.Arrays or self.Arrays[arrayname]() is None:
            array = dsa.VTKCompositeDataArray(
                dataset=self.DataSet, name=arrayname, association=self.Association
            )
            self.Arrays[arrayname] = weakref.ref(array)
        else:
            array = self.Arrays[arrayname]()
        return array
