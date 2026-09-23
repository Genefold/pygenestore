# Arrow collections: store/load pyarrow tables on the Lance backend.
# Polars and pandas DataFrames convert through pyarrow, so pyarrow.Table
# is the canonical input here.

import uuid

import numpy as np
import pyarrow as pa
import pytest

import genestore


def lstore(tmp_path):
    d = tmp_path / f"lance_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return genestore.store_array(str(d)).build()


def vector_table(rows=32, dim=8, offset=0.0):
    np.random.seed(3407)
    flat = (np.random.randn(rows * dim) + offset).astype(np.float64)
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.float64()), dim
    )
    ids = pa.array(np.arange(rows, dtype=np.uint32), type=pa.uint32())
    schema = pa.schema([
        pa.field("id", pa.uint32(), nullable=False),
        pa.field(
            "vectors",
            pa.list_(pa.field("item", pa.float64(), nullable=False), dim),
            nullable=False,
        ),
    ])
    return pa.Table.from_arrays([ids, vectors], schema=schema)


def test_store_table_and_load_table_roundtrip(tmp_path):
    table = vector_table()
    storage = lstore(tmp_path)

    path = storage.store_table(table, "vecs")
    assert isinstance(path, str)

    loaded = storage.load_table("vecs")
    assert isinstance(loaded, pa.Table)
    # The kernel writer preserves top-level nullability and stamps `kind`,
    # but does not preserve FixedSizeList child-field nullability
    # (documented conformance finding, genegraph-storage #75).
    for want, got in zip(table.schema, loaded.schema):
        assert want.name == got.name
        assert want.nullable == got.nullable
    assert loaded.schema.metadata[b"kind"] == b"vector-space"
    assert loaded.num_rows == table.num_rows
    want = table.column("vectors").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    got = loaded.column("vectors").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(want, got)


def test_store_table_accepts_pandas_dataframe(tmp_path):
    pd = pytest.importorskip("pandas")
    table = vector_table(rows=8)
    df = table.to_pandas()
    storage = lstore(tmp_path)

    storage.store_table(pa.Table.from_pandas(df, schema=table.schema), "df_vecs")
    loaded = storage.load_table("df_vecs")
    assert loaded.num_rows == 8


def test_store_table_stamps_user_properties(tmp_path):
    table = vector_table(rows=4)
    storage = lstore(tmp_path)

    storage.store_table(table, "stamped", properties={"origin": "test-suite"})
    loaded = storage.load_table("stamped")
    assert loaded.schema.metadata is not None


def test_store_table_rejects_reserved_properties(tmp_path):
    table = vector_table(rows=4)
    storage = lstore(tmp_path)

    with pytest.raises(genestore.StorageError):
        storage.store_table(table, "shadowed", properties={"filetype": "dense"})


def test_store_table_rejects_zarr_backend(tmp_path):
    d = tmp_path / f"zarr_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    z = genestore.store_array(str(d), backend="zarr").build()

    with pytest.raises(genestore.StorageError):
        z.store_table(vector_table(rows=4), "vecs")


def test_load_table_rejects_unknown_name(tmp_path):
    storage = lstore(tmp_path)
    with pytest.raises(genestore.StorageError):
        storage.load_table("missing")