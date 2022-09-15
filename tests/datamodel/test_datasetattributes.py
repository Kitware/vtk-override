import numpy as np
import pytest

from vtk_override.utils.ndarray import vtk_ndarray
from vtk_override.utils.sources import Plane


def test_copy_attributes(wavelet):
    raise NotImplementedError


@pytest.fixture()
def insert_arange_narray(wavelet):
    n_points = wavelet.point_data.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    wavelet.point_data.set_array(sample_array, "sample_array")
    return wavelet.point_data, sample_array


@pytest.fixture()
def insert_bool_array(wavelet):
    n_points = wavelet.point_data.dataset.GetNumberOfPoints()
    sample_array = np.ones(n_points, np.bool_)
    wavelet.point_data.set_array(sample_array, "sample_array")
    return wavelet.point_data, sample_array


@pytest.fixture()
def insert_string_array(wavelet):
    n_points = wavelet.point_data.dataset.GetNumberOfPoints()
    sample_array = np.repeat("A", n_points)
    wavelet.point_data.set_array(sample_array, "sample_array")
    return wavelet.point_data, sample_array


def test_point_data(wavelet):
    assert wavelet.point_data is not None
    assert "RTData" in wavelet.point_data


def test_add_point_data(cube):
    cube.clear_data()
    cube.point_data["my_array"] = np.random.random(cube.n_points)
    cube.point_data["my_other_array"] = np.arange(cube.n_points)


def test_valid_array_len_points(wavelet):
    assert wavelet.point_data.valid_array_len == wavelet.n_points


def test_valid_array_len_cells(wavelet):
    assert wavelet.cell_data.valid_array_len == wavelet.n_cells


def test_valid_array_len_field(wavelet):
    assert wavelet.field_data.valid_array_len is None


def test_get(sphere):
    point_data = np.arange(sphere.n_points)
    sphere.clear_data()
    key = "my-data"
    sphere.point_data[key] = point_data
    assert np.array_equal(sphere.point_data.get(key), point_data)
    assert sphere.point_data.get("invalid-key") is None

    default = "default"
    assert sphere.point_data.get("invalid-key", default) is default


def test_active_scalars_name(sphere):
    sphere.clear_data()
    assert sphere.point_data.active_scalars_name is None

    key = "data0"
    sphere.point_data[key] = range(sphere.n_points)
    assert sphere.point_data.active_scalars_name == key

    sphere.point_data.active_scalars_name = None
    assert sphere.point_data.active_scalars_name is None


def test_set_scalars(sphere):
    scalars = np.array(sphere.n_points)
    key = "scalars"
    sphere.point_data.set_scalars(scalars, key)
    assert sphere.point_data.active_scalars_name == key


def test_eq(sphere):
    sphere.clear_data()

    # check wrong type
    assert sphere.point_data != [1, 2, 3]

    sphere.point_data["data0"] = np.zeros(sphere.n_points)
    sphere.point_data["data1"] = np.arange(sphere.n_points)
    deep_cp = sphere.copy(deep=True)
    shal_cp = sphere.copy(deep=False)
    assert sphere.point_data == deep_cp.point_data
    assert sphere.point_data == shal_cp.point_data

    # verify inplace change
    sphere.point_data["data0"] += 1
    assert sphere.point_data != deep_cp.point_data
    assert sphere.point_data == shal_cp.point_data

    # verify key removal
    deep_cp = sphere.copy(deep=True)
    del deep_cp.point_data["data0"]
    assert sphere.point_data != deep_cp.point_data


def test_add_matrix(wavelet):
    mat_shape = (wavelet.n_points, 3, 2)
    mat = np.random.random(mat_shape)
    wavelet.point_data.set_array(mat, "mat")
    matout = wavelet.point_data["mat"].reshape(mat_shape)
    assert np.allclose(mat, matout)


def test_set_active_scalars_fail(wavelet):
    with pytest.raises(ValueError):
        wavelet.set_active_scalars("foo", preference="field")
    with pytest.raises(KeyError):
        wavelet.set_active_scalars("foo")


def test_set_active_vectors(wavelet):
    vectors = np.random.random((wavelet.n_points, 3))
    wavelet["vectors"] = vectors
    wavelet.set_active_vectors("vectors")
    assert np.allclose(wavelet.active_vectors, vectors)


def test_set_vectors(wavelet):
    assert wavelet.point_data.active_vectors is None
    vectors = np.random.random((wavelet.n_points, 3))
    wavelet.point_data.set_vectors(vectors, "my-vectors")
    assert np.allclose(wavelet.point_data.active_vectors, vectors)

    # check clearing
    wavelet.point_data.active_vectors_name = None
    assert wavelet.point_data.active_vectors_name is None


def test_set_invalid_vectors(wavelet):
    # verify non-vector data does not become active vectors
    not_vectors = np.random.random(wavelet.n_points)
    with pytest.raises(ValueError):
        wavelet.point_data.set_vectors(not_vectors, "my-vectors")


def test_set_tcoords_name(cube):
    old_name = cube.point_data.active_t_coords_name
    assert cube.point_data.active_t_coords_name is not None
    cube.point_data.active_t_coords_name = None
    assert cube.point_data.active_t_coords_name is None

    cube.point_data.active_t_coords_name = old_name
    assert cube.point_data.active_t_coords_name == old_name


#############


def test_set_bitarray(wavelet):
    """Test bitarrays are properly loaded and represented in datasetattributes."""
    wavelet.clear_data()
    assert "bool" not in str(wavelet.point_data)

    arr = np.zeros(wavelet.n_points, dtype=bool)
    arr[::2] = 1
    wavelet.point_data["bitarray"] = arr

    assert wavelet.point_data["bitarray"].dtype == np.bool_
    assert "bool" in str(wavelet.point_data)
    assert np.allclose(wavelet.point_data["bitarray"], arr)

    # ensure overwriting the type changes association
    wavelet.point_data["bitarray"] = arr.astype(np.int32)
    assert wavelet.point_data["bitarray"].dtype == np.int32


@pytest.mark.parametrize("array_key", ["invalid_array_name", -1])
def test_get_array_should_fail_if_does_not_exist(array_key, wavelet):
    with pytest.raises(KeyError):
        wavelet.point_data.get_array(array_key)


def test_get_array_should_return_bool_array(insert_bool_array):
    dsa, _ = insert_bool_array
    output_array = dsa.get_array("sample_array")
    assert output_array.dtype == np.bool_


def test_get_array_bool_array_should_be_identical(insert_bool_array):
    dsa, sample_array = insert_bool_array
    output_array = dsa.get_array("sample_array")
    assert np.array_equal(output_array, sample_array)


def test_add_should_not_add_none_array(wavelet):
    with pytest.raises(TypeError):
        wavelet.point_data.set_array(None, "sample_array")


def test_add_should_contain_array_name(insert_arange_narray):
    dsa, _ = insert_arange_narray
    assert "sample_array" in dsa


def test_add_should_contain_exact_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert np.array_equal(sample_array, dsa["sample_array"])


def test_getters_should_return_same_result(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    result_a = dsa.get_array("sample_array")
    result_b = dsa["sample_array"]
    assert np.array_equal(result_a, result_b)


def test_contains_should_contain_when_added(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert "sample_array" in dsa


def test_set_array_catch(wavelet):
    data = np.zeros(wavelet.n_points)
    with pytest.raises(TypeError, match="`name` must be a string"):
        wavelet.point_data.set_array(data, name=["foo"])


# @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
# @given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
# def test_set_array_should_accept_scalar_value(scalar, wavelet):
#     wavelet.point_data.set_array(scalar, name='int_array')


# @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
# @given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
# def test_set_array_scalar_value_should_give_array(scalar, wavelet):
#     wavelet.point_data.set_array(scalar, name='int_array')
#     expected = np.full(wavelet.point_data.dataset.n_points, scalar)
#     assert np.array_equal(expected, wavelet.point_data['int_array'])


# @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
# @given(arr=lists(text(alphabet=ascii_letters + digits + whitespace), max_size=16))
# def test_set_array_string_lists_should_equal(arr, wavelet):
#     wavelet.field_data['string_arr'] = arr
#     assert arr == wavelet.field_data['string_arr'].tolist()


# @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
# @given(arr=arrays(dtype='U', shape=10))
# def test_set_array_string_array_should_equal(arr, wavelet):
#     if not ''.join(arr).isascii():
#         with pytest.raises(ValueError, match='non-ASCII'):
#             wavelet.field_data['string_arr'] = arr
#         return

#     wavelet.field_data['string_arr'] = arr
#     assert np.array_equiv(arr, wavelet.field_data['string_arr'])


def test_wavelet_field_attributes_active_scalars(wavelet):
    with pytest.raises(TypeError):
        wavelet.field_data.active_scalars


def test_should_remove_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    dsa.remove("sample_array")
    assert "sample_array" not in dsa


def test_should_del_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    del dsa["sample_array"]
    assert "sample_array" not in dsa


def test_should_pop_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    dsa.pop("sample_array")
    assert "sample_array" not in dsa


def test_pop_should_return_arange_narray(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    other_array = dsa.pop("sample_array")
    assert np.array_equal(other_array, sample_array)


def test_pop_should_return_bool_array(insert_bool_array):
    dsa, sample_array = insert_bool_array
    other_array = dsa.pop("sample_array")
    assert np.array_equal(other_array, sample_array)


def test_pop_should_return_string_array(insert_string_array):
    dsa, sample_array = insert_string_array
    other_array = dsa.pop("sample_array")
    assert np.array_equal(other_array, sample_array)


def test_should_pop_array_invalid(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    key = "invalid_key"
    assert key not in dsa
    default = 20
    assert dsa.pop(key, default) is default


@pytest.mark.parametrize("removed_key", [None, "nonexistent_array_name", "", -1])
def test_remove_should_fail_on_bad_argument(removed_key, wavelet):
    if removed_key in [None, -1]:
        with pytest.raises(TypeError):
            wavelet.point_data.remove(removed_key)
    else:
        with pytest.raises(KeyError):
            wavelet.point_data.remove(removed_key)


@pytest.mark.parametrize("removed_key", [None, "nonexistent_array_name", "", -1])
def test_del_should_fail_bad_argument(removed_key, wavelet):
    if removed_key in [None, -1]:
        with pytest.raises(TypeError):
            del wavelet.point_data[removed_key]
    else:
        with pytest.raises(KeyError):
            del wavelet.point_data[removed_key]


@pytest.mark.parametrize("removed_key", [None, "nonexistent_array_name", "", -1])
def test_pop_should_fail_bad_argument(removed_key, wavelet):
    if removed_key in [None, -1]:
        with pytest.raises(TypeError):
            wavelet.point_data.pop(removed_key)
    else:
        with pytest.raises(KeyError):
            wavelet.point_data.pop(removed_key)


def test_length_should_increment_on_set_array(wavelet):
    initial_len = len(wavelet.point_data)
    n_points = wavelet.point_data.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    wavelet.point_data.set_array(sample_array, "sample_array")
    assert len(wavelet.point_data) == initial_len + 1


def test_length_should_decrement_on_remove(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    initial_len = len(dsa)
    dsa.remove("sample_array")
    assert len(dsa) == initial_len - 1


def test_length_should_decrement_on_pop(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    initial_len = len(dsa)
    dsa.pop("sample_array")
    assert len(dsa) == initial_len - 1


def test_length_should_be_0_on_clear(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert len(dsa) != 0
    dsa.clear()
    assert len(dsa) == 0


def test_keys_should_be_strings(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for name in dsa.keys():
        assert type(name) == str


def test_key_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert "sample_array" in dsa.keys()


def test_values_should_be_vtk_ndarrays(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        assert type(arr) == vtk_ndarray


def test_value_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        if np.array_equal(sample_array, arr):
            return
    raise AssertionError("Array not in values.")


def test_active_scalars_setter(wavelet):
    dsa = wavelet.point_data
    assert dsa.active_scalars is None

    dsa.active_scalars_name = "sample_point_scalars"
    assert dsa.active_scalars is not None
    assert dsa.GetScalars().GetName() == "sample_point_scalars"


# @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
# @given(arr=arrays(dtype='U', shape=10))
# def test_preserve_field_data_after_extract_cells(wavelet, arr):
#     if not ''.join(arr).isascii():
#         with pytest.raises(ValueError, match='non-ASCII'):
#             wavelet.field_data["foo"] = arr
#         return

#     # https://github.com/pyvista/pyvista/pull/934
#     wavelet.field_data["foo"] = arr
#     extracted = wavelet.extract_cells([0, 1, 2, 3])
#     assert "foo" in extracted.field_data


def test_assign_labels_to_points(wavelet):
    wavelet.point_data.clear()
    labels = [f"Label {i}" for i in range(wavelet.n_points)]
    wavelet["labels"] = labels
    assert (wavelet["labels"] == labels).all()


def test_normals_get(plane):
    plane.clear_data()
    assert plane.point_data.active_normals is None

    plane_w_normals = plane.compute_normals()
    assert np.array_equal(plane_w_normals.point_data.active_normals, plane_w_normals.point_normals)

    plane.point_data.active_normals_name = None
    assert plane.point_data.active_normals_name is None


def test_normals_set():
    plane = Plane(i_resolution=1, j_resolution=1)
    plane.point_data.normals = plane.point_normals
    assert np.array_equal(plane.point_data.active_normals, plane.point_normals)

    with pytest.raises(ValueError, match="must be a 2-dim"):
        plane.point_data.active_normals = [1]
    with pytest.raises(ValueError, match="must match number of points"):
        plane.point_data.active_normals = [[1, 1, 1], [0, 0, 0]]
    with pytest.raises(ValueError, match="Normals must have exactly 3 components"):
        plane.point_data.active_normals = [[1, 1], [0, 0], [0, 0], [0, 0]]


def test_normals_name(plane):
    plane.clear_data()
    assert plane.point_data.active_normals_name is None

    key = "data"
    plane.point_data.set_array(plane.point_normals, key)
    plane.point_data.active_normals_name = key
    assert plane.point_data.active_normals_name == key


def test_normals_raise_field(plane):
    with pytest.raises(AttributeError):
        plane.field_data.active_normals


def test_add_two_vectors():
    """Ensure we can add two vectors"""
    mesh = Plane(i_resolution=1, j_resolution=1)
    mesh.point_data.set_array(range(4), "my-scalars")
    mesh.point_data.set_array(range(5, 9), "my-other-scalars")
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, "vectors0")
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, "vectors1")

    assert "vectors0" in mesh.point_data
    assert "vectors1" in mesh.point_data


def test_active_vectors_name_setter():
    mesh = Plane(i_resolution=1, j_resolution=1)
    mesh.point_data.set_array(range(4), "my-scalars")
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, "vectors0")
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, "vectors1")

    assert mesh.point_data.active_vectors_name == "vectors1"
    mesh.point_data.active_vectors_name = "vectors0"
    assert mesh.point_data.active_vectors_name == "vectors0"

    with pytest.raises(KeyError, match="does not contain"):
        mesh.point_data.active_vectors_name = "not a valid key"

    with pytest.raises(ValueError, match="needs 3 components"):
        mesh.point_data.active_vectors_name = "my-scalars"


def test_active_vectors_eq():
    mesh = Plane(i_resolution=1, j_resolution=1)
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, "vectors0")
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, "vectors1")

    other_mesh = mesh.copy(deep=True)
    assert mesh == other_mesh

    mesh.point_data.active_vectors_name = "vectors0"
    assert mesh != other_mesh


def test_active_t_coords_name(plane):
    plane.point_data["arr"] = plane.point_data.active_t_coords
    plane.point_data.active_t_coords_name = "arr"

    with pytest.raises(AttributeError):
        plane.field_data.active_t_coords_name = "arr"


# @skip_windows  # windows doesn't support np.complex256
# def test_complex_raises(plane):
#     with pytest.raises(ValueError, match='Only numpy.complex64'):
#         plane.point_data['data'] = np.empty(plane.n_points, dtype=np.complex256)


@pytest.mark.parametrize("dtype_str", ["complex64", "complex128"])
def test_complex(plane, dtype_str):
    """Test if complex data can be properly represented in datasetattributes."""
    dtype = np.dtype(dtype_str)
    name = "my_data"

    with pytest.raises(ValueError, match="Complex data must be single dimensional"):
        plane.point_data[name] = np.empty((plane.n_points, 2), dtype=dtype)

    real_type = np.float32 if dtype == np.complex64 else np.float64
    data = np.random.random((plane.n_points, 2)).astype(real_type).view(dtype).ravel()
    plane.point_data[name] = data
    assert np.array_equal(plane.point_data[name], data)

    assert dtype_str in str(plane.point_data)

    # test setter
    plane.active_scalars_name = name

    # ensure that association is removed when changing datatype
    assert plane.point_data[name].dtype == dtype
    plane.point_data[name] = plane.point_data[name].real
    assert np.issubdtype(plane.point_data[name].dtype, real_type)
