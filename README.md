# pygenestore

Python bindings for [genegraph-storage](https://github.com/tuned-org-uk/genegraph-storage).
Store numpy arrays, pyarrow collections, scipy sparse matrices and graphs
at scale in Lance or Zarr format. Handles millions of rows as far as memory goes.

## Usage

Each directory is a separate storage. Pass `backend="zarr"` for the Zarr
backend; the default is Lance.

### Default Lance API (blocking)

```python
import numpy as np
import genestore

# Configure storage
builder = genestore.store_array("./lance_data")
builder.with_max_rows_per_file(500_000)
builder.with_compression("zstd")

# Build storage instance
storage = builder.build()

# Create data (2D float64 numpy array)
np.random.seed(3407)
x = np.random.randn(1000, 128).astype(np.float64)

# Store (blocking)
path = storage.store(x, "my_dataset")
print("Stored at:", path)

# Load (blocking)
y = storage.load("my_dataset")
print("Loaded shape:", y.shape)

# Verify roundtrip
assert np.array_equal(x, y)
```

### Zarr backend (append/overwrite)

```python
zstore = genestore.store_array("./zarr_data", backend="zarr").build()

zstore.store(x, "dataset")                  # rejects overwrites
start, nrows = zstore.append("dataset", more)
n = zstore.overwrite("dataset", [(5, vector)])
entries = zstore.list_datasets()            # shape, dtype, path per dataset
summary = zstore.summary("dataset")
```

Appends serialize per dataset; concurrent appends from other processes
surface `genestore.StorageError` naming the lock file.

### Arrow collections (polars and pandas friendly)

`store_table` accepts a pyarrow `Table` or `RecordBatch`; vector-space
schemas require non-nullable columns. Polars and pandas DataFrames
convert to pyarrow Tables with zero copy.

```python
import pyarrow as pa

t = pa.table({"id": ids, "vectors": fixed_size_list_array})
storage.store_table(t, "vecs", properties={"origin": "ingest"})
loaded = storage.load_table("vecs")   # pa.Table
```

### Sparse matrices (scipy)

Sparse collections live on the Lance backend. Keys use the kernel's
logical artifact vocabulary: `adjacency`, `laplacian`, `signals`.

```python
m = csr_array(x)
storage.store_sparse(m.indptr, m.indices, m.data, m.shape, "adjacency")
indptr, indices, data, shape = storage.load_sparse("adjacency")
rebuilt = csr_array((data, indices, indptr), shape=shape)
```

### Graphs

```python
storage.store_graph("g", src, dst, weights=w, num_nodes=4)
g = storage.load_graph("g")   # src, dst, weight arrays + facts
```

Pass `node_id_width="u64"` when ids exceed `u32::MAX`.

### Async API

```python
import asyncio

async def main():
    builder = genestore.create_storage("./lance_data")
    storage = builder.build()
    path = await storage.aio.store(x, "my_dataset")
    y = await storage.aio.load("my_dataset")
    start, nrows = await storage.aio.append("dataset", more)  # zarr backend

asyncio.run(main())
```

## Errors

All storage failures raise `genestore.StorageError`. Kernel error
variants (invalid data, unsupported filetype, lock contention) surface
through it with the kernel's message.

## Tests

The dev environment is managed with uv and declared in `pyproject.toml`:

```bash
uv sync --group dev
uv run maturin develop --release
uv run pytest tests/
```