import weakref

from vtkmodules.numpy_interface import dataset_adapter as dsa

from vtk_override.datamodel.object import DataObject
from vtk_override.fields import CompositeDataSetAttributes


class CompositeDataIterator(object):
    """Wrapper for a vtkCompositeDataIterator class to satisfy
    the python iterator protocol. This iterator iterates
    over non-empty leaf nodes. To iterate over empty or
    non-leaf nodes, use the vtkCompositeDataIterator directly.
    """

    def __init__(self, cds):
        self.Iterator = cds.NewIterator()
        if self.Iterator:
            self.Iterator.UnRegister(None)
            self.Iterator.GoToFirstItem()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.Iterator:
            raise StopIteration

        if self.Iterator.IsDoneWithTraversal():
            raise StopIteration
        retVal = self.Iterator.GetCurrentDataObject()
        self.Iterator.GoToNextItem()
        return retVal

    def next(self):
        return self.__next__()

    def __getattr__(self, name):
        """Returns attributes from the vtkCompositeDataIterator."""
        return getattr(self.Iterator, name)


class CompositeDataSetBase(object):
    """A wrapper for vtkCompositeData and subclasses that makes it easier
    to access Point/Cell/Field data as VTKCompositeDataArrays. It also
    provides a Python type iterator."""

    def __init__(self):
        self._PointData = None
        self._CellData = None
        self._FieldData = None
        self._Points = None

    def __iter__(self):
        "Creates an iterator for the contained datasets."
        return CompositeDataIterator(self)

    def get_attributes(self, type):
        """Returns the attributes specified by the type as a
        CompositeDataSetAttributes instance."""
        return CompositeDataSetAttributes(self, type)

    @property
    def point_data(self):
        "Returns the point data as a DataSetAttributes instance."
        if self._PointData is None or self._PointData() is None:
            pdata = self.get_attributes(DataObject.POINT)
            self._PointData = weakref.ref(pdata)
        return self._PointData()

    @property
    def cell_data(self):
        "Returns the cell data as a DataSetAttributes instance."
        if self._CellData is None or self._CellData() is None:
            cdata = self.get_attributes(DataObject.CELL)
            self._CellData = weakref.ref(cdata)
        return self._CellData()

    @property
    def field_data(self):
        "Returns the field data as a DataSetAttributes instance."
        if self._FieldData is None or self._FieldData() is None:
            fdata = self.get_attributes(DataObject.FIELD)
            self._FieldData = weakref.ref(fdata)
        return self._FieldData()

    @property
    def points(self):
        "Returns the points as a VTKCompositeDataArray instance."
        if self._Points is None or self._Points() is None:
            pts = []
            for ds in self:
                try:
                    _pts = ds.Points
                except AttributeError:
                    _pts = None

                if _pts is None:
                    pts.append(dsa.NoneArray)
                else:
                    pts.append(_pts)
            if len(pts) == 0 or all([a is dsa.NoneArray for a in pts]):
                cpts = dsa.NoneArray
            else:
                cpts = dsa.VTKCompositeDataArray(pts, dataset=self)
            self._Points = weakref.ref(cpts)
        return self._Points()
