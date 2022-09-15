from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet

from vtk_override.composite.composite import CompositeDataSetBase
from vtk_override.utils import override


@override(vtkPartitionedDataSet)
class PartitionedDataSet(vtkPartitionedDataSet, CompositeDataSetBase):
    def append(self, dataset):
        self.SetPartition(self.GetNumberOfPartitions(), dataset)
