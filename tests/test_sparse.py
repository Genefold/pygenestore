# Sparse matrices: scipy CSR arrays in, scipy CSR arrays out, on the Lance
# backend. Zarr rejects sparse collections with a typed error.

import uuid

import numpy as np
import pytest
from scipy.sparse import csr_array

import genestore


def lstore(tmp_path):
    d = tmp_path / f"lance_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return genestore.store_array(str(d)).build()


def sparse_fixture():
    np.random.seed(3407)
    dense = np.zeros((20, 30), dtype=np.float64)
    for _ in range(40):
        r, c = np.random.randint(0, 20), np.random.randint(0, 30)
        dense[r, c] = np.random.rand()
    return csr_array(dense)


def test_store_sparse_load_sparse_roundtrip(tmp_path):
    m = sparse_fixture()
    storage = lstore(tmp_path)

    path = storage.store_sparse(m.indptr, m.indices, m.data, m.shape, "adjacency")
    assert isinstance(path, str)

    indptr, indices, data, shape = storage.load_sparse("adjacency")
    rebuilt = csr_array((data, indices, indptr), shape=shape)
    assert shape == m.shape
    assert np.array_equal(rebuilt.toarray(), m.toarray())


def test_load_sparse_matches_scipy_layout(tmp_path):
    m = sparse_fixture()
    storage = lstore(tmp_path)
    storage.store_sparse(m.indptr, m.indices, m.data, m.shape, "adjacency")

    indptr, indices, data, shape = storage.load_sparse("adjacency")
    assert np.array_equal(np.asarray(indptr), m.indptr)
    assert np.array_equal(np.asarray(indices), m.indices)
    assert np.array_equal(np.asarray(data), m.data)


def test_store_sparse_rejects_empty_matrix(tmp_path):
    m = csr_array((5, 5), dtype=np.float64)
    storage = lstore(tmp_path)

    with pytest.raises(genestore.StorageError):
        storage.store_sparse(m.indptr, m.indices, m.data, m.shape, "adjacency")


def test_store_sparse_rejects_zarr_backend(tmp_path):
    d = tmp_path / f"zarr_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    z = genestore.store_array(str(d), backend="zarr").build()
    m = sparse_fixture()

    with pytest.raises(genestore.StorageError):
        z.store_sparse(m.indptr, m.indices, m.data, m.shape, "adjacency")