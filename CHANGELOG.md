# Changelog

## 0.72.0 (2026-09-23)

Rebound to genegraph-storage 0.72.0 (was 0.10.0). Breaking release: the
Python surface changed with the kernel, and stores written by the 0.10
binding (official Lance 6 format) are not supported.

### Added

- Zarr backend (`backend="zarr"`): `store`, `load`, `append`
  (`append_vectors` offsets), `overwrite` (row updates),
  `list_datasets`, `summary`. Registry seeded on first store.
- Arrow collections on Lance: `store_table` (pyarrow Table or
  RecordBatch; single batch; non-nullable vector-space schema) and
  `load_table` returning a pyarrow Table with the kernel's `kind` stamp.
- Sparse collections on Lance: `store_sparse(indptr, indices, data,
  shape, key)` and `load_sparse(key)`; keys follow the kernel's logical
  artifact vocabulary (`adjacency`, `laplacian`, `signals`).
- Graph collections on Lance: `store_graph(name, src, dst, weights,
  num_nodes, node_id_width)` and `load_graph(name)` over numpy arrays.
- `genestore.StorageError` wrapping all kernel `StorageError` variants.
- Dev dependencies declared as a uv dependency group in
  `pyproject.toml`; `requirements-dev.txt` removed.

### Changed

- **Breaking**: `LanceStorage`/`LanceStorageAsync` classes replaced by
  `Storage`/`StorageAsync` over both backends; `store_array(output_dir,
  backend=..., label=...)` selects the backend, `create_storage` stays
  as alias.
- **Breaking**: `load` no longer falls back to the `rawinput` key.
- Dependency stack: pyo3 0.29, numpy 0.29, smartcore 0.6.14, sprs
  0.11.5, arrow 60 (aligned with genegraph-storage 0.72).
- Relative output directories are canonicalized at `build()`; the
  kernel rejects relative paths.
- Builder `with_max_rows_per_file`/`with_max_rows_per_group`/
  `with_compression` are reserved knobs: the kernel manages lancefmt
  fragmentation internally and does not expose writer options yet.
- CI drops protoc setup (the kernel vendors protobuf types) and gains a
  test job running the suite on the built wheel.