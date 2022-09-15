def test_unstructured_grid_eq(hexbeam):
    copy = hexbeam.copy()
    assert hexbeam == copy

    copy = hexbeam.copy()
    hexbeam.celltypes[0] = 0
    assert hexbeam != copy

    # TODO: this isn't working
    # copy = hexbeam.copy()
    # hexbeam.cell_connectivity[0] += 1
    # assert hexbeam != copy
