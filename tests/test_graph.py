# Graph collections: numpy edge arrays in, numpy edge arrays out, on the
# Lance backend. Graph storage on Zarr is a typed rejection upstream.

import uuid

import numpy as np
import pytest

import genestore


def lstore(tmp_path):
    d = tmp_path / f"lance_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return genestore.store_array(str(d)).build()


def test_store_graph_load_graph_weighted_roundtrip(tmp_path):
    src = np.array([0, 1, 2, 3], dtype=np.uint64)
    dst = np.array([1, 2, 3, 0], dtype=np.uint64)
    weights = np.array([0.5, 0.25, 1.0, 0.75], dtype=np.float64)

    storage = lstore(tmp_path)
    path = storage.store_graph("g", src, dst, weights=weights, num_nodes=4)
    assert isinstance(path, str)

    g = storage.load_graph("g")
    assert g["weighted"] is True
    assert g["num_nodes"] == 4
    assert np.array_equal(np.asarray(g["src"]), src)
    assert np.array_equal(np.asarray(g["dst"]), dst)
    assert np.array_equal(np.asarray(g["weight"]), weights)


def test_store_graph_load_graph_unweighted_roundtrip(tmp_path):
    src = np.array([0, 1, 2], dtype=np.uint64)
    dst = np.array([1, 2, 0], dtype=np.uint64)

    storage = lstore(tmp_path)
    storage.store_graph("ug", src, dst, num_nodes=3)

    g = storage.load_graph("ug")
    assert g["weighted"] is False
    assert g["weight"] is None
    assert np.array_equal(np.asarray(g["src"]), src)
    assert np.array_equal(np.asarray(g["dst"]), dst)


def test_store_graph_rejects_negative_node_ids(tmp_path):
    src = np.array([0, -1], dtype=np.int64)
    dst = np.array([1, 2], dtype=np.uint64)

    storage = lstore(tmp_path)
    with pytest.raises(genestore.StorageError):
        storage.store_graph("neg", src, dst, num_nodes=2)


def test_store_graph_rejects_ids_beyond_u32_without_wide_width(tmp_path):
    big = 2**32 + 5
    src = np.array([0, big], dtype=np.uint64)
    dst = np.array([1, 2], dtype=np.uint64)

    storage = lstore(tmp_path)
    with pytest.raises(genestore.StorageError):
        storage.store_graph("g", src, dst, num_nodes=big + 1)

    path = storage.store_graph("g", src, dst, num_nodes=big + 1, node_id_width="u64")
    assert isinstance(path, str)

    g = storage.load_graph("g")
    assert g["node_id_width"] == "u64"
    assert np.asarray(g["src"])[1] == big


def test_store_graph_rejects_zarr_backend(tmp_path):
    d = tmp_path / f"zarr_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    z = genestore.store_array(str(d), backend="zarr").build()

    with pytest.raises(genestore.StorageError):
        z.store_graph("g", np.array([0], dtype=np.uint64), np.array([1], dtype=np.uint64))