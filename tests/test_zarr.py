# Zarr backend: store/load, append, overwrite, discovery.
# Spec for the Zarr backend surface of genestore (kernel 0.72).

import uuid

import numpy as np
import pytest

import genestore


def zstore(tmp_path):
    d = tmp_path / f"zarr_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return genestore.store_array(str(d), backend="zarr").build()


def test_zarr_store_load_roundtrip_blocking(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(256, 64).astype(np.float64)

    z = zstore(tmp_path)
    key = z.store(x, "dataset")

    assert isinstance(key, str)
    y = z.load("dataset")
    assert y.shape == (256, 64)
    assert np.array_equal(x, y)


def test_zarr_store_rejects_overwrite_with_storage_error(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(16, 8).astype(np.float64)

    z = zstore(tmp_path)
    z.store(x, "dataset")

    with pytest.raises(genestore.StorageError):
        z.store(x, "dataset")


def test_zarr_append_vectors_returns_offsets_and_preserves_rows(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(32, 16).astype(np.float64)
    more = np.random.randn(16, 16).astype(np.float64)

    z = zstore(tmp_path)
    z.store(x, "growing")

    start, nrows = z.append("growing", more)
    assert start == 32
    assert nrows == 48

    y = z.load("growing")
    assert y.shape == (48, 16)
    assert np.array_equal(y[:32], x)
    assert np.array_equal(y[32:], more)


def test_zarr_append_rejects_missing_dataset(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(4, 4).astype(np.float64)

    z = zstore(tmp_path)
    with pytest.raises(genestore.StorageError):
        z.append("fresh", x)


def test_zarr_overwrite_vectors_replaces_rows(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(16, 8).astype(np.float64)
    v5 = np.arange(8, dtype=np.float64) + 100.0
    v0 = np.arange(8, dtype=np.float64) + 200.0

    z = zstore(tmp_path)
    z.store(x, "patched")
    n = z.overwrite("patched", [(5, v5), (0, v0)])

    assert n == 2
    y = z.load("patched")
    assert np.array_equal(y[0], v0)
    assert np.array_equal(y[5], v5)
    assert np.array_equal(y[1:5], x[1:5])
    assert np.array_equal(y[6:], x[6:])


def test_zarr_list_datasets_reports_shape_and_dtype(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(16, 8).astype(np.float64)
    w = np.random.randn(8, 8).astype(np.float64)

    z = zstore(tmp_path)
    z.store(x, "one")
    z.store(w, "two")

    entries = {e["dataset_id"].split("--")[-1]: e for e in z.list_datasets()}
    assert "one" in entries and "two" in entries
    assert list(entries["one"]["shape"]) == [16, 8]
    assert entries["one"]["dtype"] == "float64"


def test_zarr_summary_reports_registered_shape(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(16, 8).astype(np.float64)
    v = np.random.randn(3, 8).astype(np.float64)

    z = zstore(tmp_path)
    z.store(x, "base")
    z.append("base", v)

    summary = z.summary("base")
    assert list(summary["shape"]) == [19, 8]


def test_zarr_store_rejects_traversal_keys(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(4, 4).astype(np.float64)

    z = zstore(tmp_path)
    with pytest.raises(genestore.StorageError):
        z.store(x, "../escape")
    with pytest.raises(genestore.StorageError):
        z.store(x, "a--b")


@pytest.mark.asyncio
async def test_zarr_store_load_roundtrip_async(tmp_path):
    np.random.seed(3407)
    x = np.random.randn(64, 32).astype(np.float64)

    z = zstore(tmp_path)
    key = await z.aio.store(x, "async_dataset")
    assert isinstance(key, str)

    y = await z.aio.load("async_dataset")
    assert y.shape == (64, 32)
    assert np.array_equal(x, y)