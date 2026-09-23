use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Once};

use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use genegraph_storage::StorageError as KernelError;
use genegraph_storage::graph::{GraphEdge, GraphWriteOptions, NodeIdWidth, StoredGraph};
use genegraph_storage::lance_storage_graph::LanceStorageGraph;
use genegraph_storage::metadata::GeneMetadata;
use genegraph_storage::traits::backend::StorageBackend;
use genegraph_storage::traits::metadata::Metadata as _;
use genegraph_storage::traits::zarr::{DatasetSummary, RowUpdate, ZarrStorageOps};
use genegraph_storage::zarr_storage::{ZarrStorage, decode_dataset_id, make_dataset_id};
use numpy::{PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pyo3_async_runtimes::tokio::{future_into_py, get_runtime};
use smartcore::linalg::basic::arrays::Array as _;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .ok();
    });
}

pyo3::create_exception!(genestore, StorageError, PyException);

fn pyerr(e: KernelError) -> PyErr {
    StorageError::new_err(format!("{e}"))
}

fn dense_from_parts(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<DenseMatrix<f64>> {
    DenseMatrix::new(rows, cols, data, false)
        .map_err(|e| PyValueError::new_err(format!("invalid matrix: {e}")))
}

fn numpy_to_dense(array: &PyArrayLike2<'_, f64>) -> PyResult<(usize, usize, Vec<f64>)> {
    let view = array.as_array();
    let (rows, cols) = (view.nrows(), view.ncols());
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(
            "Cannot store empty array. Array must have non-zero dimensions.",
        ));
    }
    let mut nan = 0usize;
    let mut inf = 0usize;
    for &v in view {
        if v.is_nan() {
            nan += 1;
        } else if v.is_infinite() {
            inf += 1;
        }
    }
    if nan + inf > 0 {
        return Err(PyValueError::new_err(format!(
            "Array contains non-finite values ({nan} NaN, {inf} Inf/-Inf). \
Only arrays with finite values can be stored as embeddings. \
Please clean your data before storing."
        )));
    }
    let data: Vec<f64> = view.iter().copied().collect();
    Ok((rows, cols, data))
}

fn dense_to_flat(matrix: &DenseMatrix<f64>) -> (usize, usize, Vec<f64>) {
    let (rows, cols) = matrix.shape();
    let mut row_major = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            row_major[r * cols + c] = *matrix.get((r, c));
        }
    }
    (rows, cols, row_major)
}

fn flat_to_numpy(py: Python<'_>, rows: usize, cols: usize, flat: &[f64]) -> PyResult<Py<PyAny>> {
    let view = numpy::ndarray::ArrayView2::from_shape((rows, cols), flat)
        .map_err(|e| PyValueError::new_err(format!("reshape failed: {e}")))?;
    // SAFETY: PyArray2::new allocates uninitialized memory; the assign below
    // writes every element before the array is visible to Python.
    let arr = unsafe { PyArray2::new(py, [rows, cols], false) };
    unsafe { arr.as_array_mut() }.assign(&view);
    Ok(arr.into_any().unbind())
}

fn node_ids(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<u64>> {
    if let Ok(a) = obj.extract::<PyArrayLike1<'_, u64>>() {
        return Ok(a.as_array().iter().copied().collect());
    }
    let a: PyArrayLike1<'_, i64> = obj
        .extract()
        .map_err(|_| PyValueError::new_err(format!("{field} must be an integer numpy array")))?;
    a.as_array()
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(StorageError::new_err(format!(
                    "{field} contains a negative node id ({v}); node ids are unsigned"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect()
}

async fn lance_open_or_seed(
    base: &std::path::Path,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<LanceStorageGraph, KernelError> {
    let base_s = base.to_string_lossy().to_string();
    if LanceStorageGraph::exists(&base_s).0 {
        Ok(LanceStorageGraph::spawn(base_s).await?.0)
    } else {
        let storage = LanceStorageGraph::new(base_s, name.to_string())?;
        GeneMetadata::seed_metadata(name, rows, cols, &storage).await?;
        Ok(storage)
    }
}

async fn lance_open(base: &std::path::Path) -> PyResult<LanceStorageGraph> {
    let base_s = base.to_string_lossy().to_string();
    if !LanceStorageGraph::exists(&base_s).0 {
        return Err(StorageError::new_err(format!(
            "storage at {} is not initialized; store an array first",
            base.display()
        )));
    }
    Ok(LanceStorageGraph::spawn(base_s).await.map_err(pyerr)?.0)
}

#[derive(Clone)]
struct LanceInner {
    base: PathBuf,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    compression: String,
}

impl LanceInner {
    async fn store_core(
        &self,
        name: String,
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    ) -> PyResult<String> {
        let storage = lance_open_or_seed(&self.base, &name, rows, cols)
            .await
            .map_err(pyerr)?;
        let matrix = dense_from_parts(rows, cols, data)?;
        let md_path = storage.metadata_path();
        storage
            .save_dense(&name, &matrix, &md_path)
            .await
            .map_err(pyerr)?;
        Ok(md_path.to_string_lossy().to_string())
    }

    async fn load_core(&self, name: String) -> PyResult<(usize, usize, Vec<f64>)> {
        let base_s = self.base.to_string_lossy().to_string();
        if !LanceStorageGraph::exists(&base_s).0 {
            return Err(StorageError::new_err(format!(
                "storage at {} is not initialized; store an array first",
                self.base.display()
            )));
        }
        let storage = LanceStorageGraph::spawn(base_s).await.map_err(pyerr)?.0;
        let matrix = storage.load_dense(&name).await.map_err(pyerr)?;
        Ok(dense_to_flat(&matrix))
    }
}

#[derive(Clone)]
enum Backend {
    Lance(LanceInner),
    Zarr {
        storage: Arc<ZarrStorage>,
        label: String,
        root: PathBuf,
    },
}

impl Backend {
    fn zarr_label(&self) -> String {
        match self {
            Backend::Zarr { label, .. } => label.clone(),
            _ => String::new(),
        }
    }
}

#[pyclass]
struct Storage {
    backend: Backend,
}

impl Storage {
    fn make_dataset_id(&self, name: &str) -> String {
        match &self.backend {
            Backend::Lance(_) => name.to_string(),
            Backend::Zarr { label, .. } => {
                let (l, rel) = decode_dataset_id(name);
                if l == *label {
                    format!("{label}--{rel}")
                } else {
                    make_dataset_id(label, name)
                }
            }
        }
    }

    fn dataset_key(&self, name: &str) -> String {
        match &self.backend {
            Backend::Lance(_) => name.to_string(),
            Backend::Zarr { label, .. } => {
                let (l, rel) = decode_dataset_id(name);
                if l == *label { rel } else { name.to_string() }
            }
        }
    }
}

#[pymethods]
impl Storage {
    #[pyo3(signature = (array, name))]
    fn store(
        &self,
        py: Python<'_>,
        array: PyArrayLike2<'_, f64>,
        name: String,
    ) -> PyResult<String> {
        let (rows, cols, data) = numpy_to_dense(&array)?;
        match &self.backend {
            Backend::Lance(inner) => {
                py.detach(|| get_runtime().block_on(inner.store_core(name, rows, cols, data)))
            }
            Backend::Zarr { storage, root, .. } => {
                let z = storage.clone();
                let root = root.clone();
                let key = self.dataset_key(&name);
                let id = self.make_dataset_id(&name);
                let label = self.backend.zarr_label().to_string();
                py.detach(|| {
                    get_runtime().block_on(async move {
                        if !z.metadata_path().exists() {
                            GeneMetadata::seed_metadata(&label, rows, cols, &*z)
                                .await
                                .map_err(pyerr)?;
                        }
                        let matrix = dense_from_parts(rows, cols, data)?;
                        let md = z.metadata_path();
                        z.save_dense(&key, &matrix, &md).await.map_err(pyerr)?;
                        Ok(format!("{}/{id}", root.display()))
                    })
                })
            }
        }
    }

    #[pyo3(signature = (name))]
    fn load(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(inner) => {
                let inner = inner.clone();
                let (rows, cols, flat) =
                    py.detach(|| get_runtime().block_on(inner.load_core(name)))?;
                flat_to_numpy(py, rows, cols, &flat)
            }
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let key = self.dataset_key(&name);
                let (rows, cols, flat) = py.detach(|| {
                    get_runtime().block_on(async move {
                        let matrix = z.load_dense(&key).await.map_err(pyerr)?;
                        PyResult::Ok(dense_to_flat(&matrix))
                    })
                })?;
                flat_to_numpy(py, rows, cols, &flat)
            }
        }
    }

    #[pyo3(signature = (name, array))]
    fn append(
        &self,
        py: Python<'_>,
        name: String,
        array: PyArrayLike2<'_, f64>,
    ) -> PyResult<(usize, usize)> {
        let (rows, cols, data) = numpy_to_dense(&array)?;
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "append_vectors is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let id = self.make_dataset_id(&name);
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let matrix = dense_from_parts(rows, cols, data)?;
                        z.append_vectors(&id, &matrix).await.map_err(pyerr)
                    })
                })
            }
        }
    }

    #[pyo3(signature = (name, updates))]
    fn overwrite(
        &self,
        py: Python<'_>,
        name: String,
        updates: Vec<(usize, PyArrayLike1<'_, f64>)>,
    ) -> PyResult<usize> {
        let updates: Vec<RowUpdate> = updates
            .into_iter()
            .map(|(row_index, vector)| RowUpdate {
                row_index,
                vector: vector.as_array().iter().copied().collect(),
            })
            .collect();
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "overwrite_vectors is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let id = self.make_dataset_id(&name);
                py.detach(|| {
                    get_runtime().block_on(async move {
                        z.overwrite_vectors(&id, &updates).await.map_err(pyerr)
                    })
                })
            }
        }
    }

    fn list_datasets(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "list_datasets is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let entries = py.detach(|| {
                    get_runtime().block_on(async move { z.list_datasets().await.map_err(pyerr) })
                })?;
                let dicts: Vec<Bound<'_, PyDict>> = entries
                    .iter()
                    .map(|s| summary_dict(py, s))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, dicts)?.into_any().unbind())
            }
        }
    }

    #[pyo3(signature = (name))]
    fn summary(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "summary is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let id = self.make_dataset_id(&name);
                let summary = py.detach(|| {
                    get_runtime()
                        .block_on(async move { z.summarize_by_id(&id).await.map_err(pyerr) })
                })?;
                Ok(summary_dict(py, &summary)?.into_any().unbind())
            }
        }
    }

    #[pyo3(signature = (table, name, properties=None))]
    fn store_table(
        &self,
        py: Python<'_>,
        table: &Bound<'_, PyAny>,
        name: String,
        properties: Option<BTreeMap<String, String>>,
    ) -> PyResult<String> {
        let batch = table_to_batch(table)?;
        let props: BTreeMap<String, String> = properties.unwrap_or_default();
        match &self.backend {
            Backend::Lance(inner) => {
                let rows = batch.num_rows();
                let name2 = name.clone();
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open_or_seed(&inner.base, &name2, rows, 0)
                            .await
                            .map_err(pyerr)?;
                        let md_path = storage.metadata_path();
                        storage
                            .save_vectors_with(&name, &batch, &props, &md_path)
                            .await
                            .map_err(pyerr)?;
                        Ok(storage.file_path(&name).to_string_lossy().to_string())
                    })
                })
            }
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let key = self.dataset_key(&name);
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let props = BTreeMap::new();
                        let md = z.metadata_path();
                        z.save_vectors_with(&key, &batch, &props, &md)
                            .await
                            .map_err(pyerr)?;
                        Ok(key)
                    })
                })
            }
        }
    }

    fn load_table(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(inner) => {
                let base = inner.base.clone();
                let batch = py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open(&base).await?;
                        storage.load_vectors(&name).await.map_err(pyerr)
                    })
                })?;
                record_batch_to_py(py, batch)
            }
            Backend::Zarr { .. } => Err(StorageError::new_err(
                "vector collections are not supported on the zarr backend",
            )),
        }
    }

    #[pyo3(signature = (indptr, indices, data, shape, name))]
    fn store_sparse(
        &self,
        py: Python<'_>,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        data: PyArrayLike1<'_, f64>,
        shape: (usize, usize),
        name: String,
    ) -> PyResult<String> {
        let indptr = index_list(indptr, "indptr")?;
        let indices = index_list(indices, "indices")?;
        let matrix = csr_from_parts(indptr, indices, &data, shape)?;
        match &self.backend {
            Backend::Lance(inner) => {
                let (nrows, ncols) = shape;
                let name2 = name.clone();
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open_or_seed(&inner.base, &name2, nrows, ncols)
                            .await
                            .map_err(pyerr)?;
                        let md_path = storage.metadata_path();
                        storage
                            .save_sparse(&name, &matrix, &md_path)
                            .await
                            .map_err(pyerr)?;
                        Ok(storage.file_path(&name).to_string_lossy().to_string())
                    })
                })
            }
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let key = self.dataset_key(&name);
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let md = z.metadata_path();
                        z.save_sparse(&key, &matrix, &md).await.map_err(pyerr)?;
                        Ok(key)
                    })
                })
            }
        }
    }

    #[pyo3(signature = (name))]
    fn load_sparse(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(inner) => {
                let base = inner.base.clone();
                let matrix = py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open(&base).await?;
                        storage.load_sparse(&name).await.map_err(pyerr)
                    })
                })?;
                sparse_to_py(py, &matrix)
            }
            Backend::Zarr { .. } => Err(StorageError::new_err(
                "sparse collections are not supported on the zarr backend",
            )),
        }
    }

    #[pyo3(signature = (name, src, dst, weights=None, num_nodes=None, node_id_width=None))]
    #[allow(clippy::too_many_arguments)]
    fn store_graph(
        &self,
        py: Python<'_>,
        name: String,
        src: &Bound<'_, PyAny>,
        dst: &Bound<'_, PyAny>,
        weights: Option<PyArrayLike1<'_, f64>>,
        num_nodes: Option<u64>,
        node_id_width: Option<String>,
    ) -> PyResult<String> {
        let (edges, options) = build_graph(src, dst, weights, num_nodes, node_id_width)?;
        let rows_hint = num_nodes.unwrap_or(0) as usize;
        match &self.backend {
            Backend::Lance(inner) => {
                let name2 = name.clone();
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open_or_seed(&inner.base, &name2, rows_hint, 0)
                            .await
                            .map_err(pyerr)?;
                        let md_path = storage.metadata_path();
                        storage
                            .save_graph_with(&name, &edges, &options, &md_path)
                            .await
                            .map_err(pyerr)?;
                        Ok(storage.file_path(&name).to_string_lossy().to_string())
                    })
                })
            }
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let key = self.dataset_key(&name);
                py.detach(|| {
                    get_runtime().block_on(async move {
                        let md = z.metadata_path();
                        z.save_graph_with(&key, &edges, &options, &md)
                            .await
                            .map_err(pyerr)?;
                        Ok(key)
                    })
                })
            }
        }
    }

    fn load_graph(&self, py: Python<'_>, name: String) -> PyResult<Py<PyAny>> {
        match &self.backend {
            Backend::Lance(inner) => {
                let base = inner.base.clone();
                let graph = py.detach(|| {
                    get_runtime().block_on(async move {
                        let storage = lance_open(&base).await?;
                        storage.load_graph(&name).await.map_err(pyerr)
                    })
                })?;
                graph_to_py(py, graph)
            }
            Backend::Zarr { .. } => Err(StorageError::new_err(
                "graph collections are not supported on the zarr backend",
            )),
        }
    }

    #[getter]
    fn aio(&self) -> StorageAsync {
        StorageAsync {
            backend: self.backend.clone(),
        }
    }

    fn get_config(&self) -> String {
        match &self.backend {
            Backend::Lance(inner) => format!(
                "LanceStorage(output_dir='{}', max_rows_per_file={}, max_rows_per_group={}, compression='{}')",
                inner.base.display(),
                inner.max_rows_per_file,
                inner.max_rows_per_group,
                inner.compression
            ),
            Backend::Zarr { root, label, .. } => {
                format!("ZarrStorage(root='{}', label='{}')", root.display(), label)
            }
        }
    }

    fn get_output_dir(&self) -> String {
        match &self.backend {
            Backend::Lance(inner) => inner.base.to_string_lossy().to_string(),
            Backend::Zarr { root, .. } => root.to_string_lossy().to_string(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.backend {
            Backend::Lance(inner) => {
                format!("LanceStorage(output_dir='{}')", inner.base.display())
            }
            Backend::Zarr { root, label, .. } => {
                format!("ZarrStorage(root='{}', label='{}')", root.display(), label)
            }
        }
    }
}

/// Storage builder for configuring backend parameters.
#[pyclass]
struct StorageBuilder {
    output_dir: PathBuf,
    backend: String,
    label: Option<String>,
    max_rows_per_file: Option<usize>,
    max_rows_per_group: Option<usize>,
    compression: Option<String>,
}

#[pymethods]
impl StorageBuilder {
    #[new]
    #[pyo3(signature = (output_dir, backend=None, label=None, max_rows_per_file=None, max_rows_per_group=None, compression=None))]
    fn new(
        output_dir: String,
        backend: Option<String>,
        label: Option<String>,
        max_rows_per_file: Option<usize>,
        max_rows_per_group: Option<usize>,
        compression: Option<String>,
    ) -> PyResult<Self> {
        let backend = backend.unwrap_or_else(|| "lance".to_string());
        if backend != "lance" && backend != "zarr" {
            return Err(PyValueError::new_err(format!(
                "unknown backend '{backend}': expected 'lance' or 'zarr'"
            )));
        }
        Ok(Self {
            output_dir: PathBuf::from(output_dir),
            backend,
            label,
            max_rows_per_file,
            max_rows_per_group,
            compression,
        })
    }

    fn with_output_dir(&mut self, output_dir: String) -> PyResult<()> {
        self.output_dir = PathBuf::from(output_dir);
        Ok(())
    }

    fn with_max_rows_per_file(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_file = Some(max_rows);
        Ok(())
    }

    fn with_max_rows_per_group(&mut self, max_rows: usize) -> PyResult<()> {
        self.max_rows_per_group = Some(max_rows);
        Ok(())
    }

    fn with_compression(&mut self, compression: String) -> PyResult<()> {
        self.compression = Some(compression);
        Ok(())
    }

    fn with_label(&mut self, label: String) -> PyResult<()> {
        self.label = Some(label);
        Ok(())
    }

    fn build(&self) -> PyResult<Storage> {
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| PyException::new_err(format!("Failed to create directory: {e}")))?;
        let root = self
            .output_dir
            .canonicalize()
            .map_err(|e| PyException::new_err(format!("Failed to resolve directory: {e}")))?;
        match self.backend.as_str() {
            "zarr" => {
                let label = self.label.clone().unwrap_or_else(|| {
                    root.file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "genestore".to_string())
                });
                let storage = ZarrStorage::new(&root, &label).map_err(pyerr)?;
                Ok(Storage {
                    backend: Backend::Zarr {
                        storage: Arc::new(storage),
                        label,
                        root: root.clone(),
                    },
                })
            }
            _ => Ok(Storage {
                backend: Backend::Lance(LanceInner {
                    base: root,
                    max_rows_per_file: self.max_rows_per_file.unwrap_or(1_000_000),
                    max_rows_per_group: self.max_rows_per_group.unwrap_or(10_000),
                    compression: self
                        .compression
                        .clone()
                        .unwrap_or_else(|| "zstd".to_string()),
                }),
            }),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageBuilder(output_dir='{}', backend='{}')",
            self.output_dir.display(),
            self.backend
        )
    }
}

/// Async facade exposed at `storage.aio`.
#[pyclass]
struct StorageAsync {
    backend: Backend,
}

#[pymethods]
impl StorageAsync {
    #[pyo3(signature = (array, name))]
    fn store<'py>(
        &self,
        py: Python<'py>,
        array: PyArrayLike2<'_, f64>,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (rows, cols, data) = numpy_to_dense(&array)?;
        match &self.backend {
            Backend::Lance(inner) => {
                let inner = inner.clone();
                future_into_py(
                    py,
                    async move { inner.store_core(name, rows, cols, data).await },
                )
            }
            Backend::Zarr { storage, root, .. } => {
                let z = storage.clone();
                let id = Storage {
                    backend: self.backend.clone(),
                }
                .make_dataset_id(&name);
                let key = Storage {
                    backend: self.backend.clone(),
                }
                .dataset_key(&name);
                let root = root.clone();
                let label = self.backend.zarr_label();
                future_into_py(py, async move {
                    if !z.metadata_path().exists() {
                        GeneMetadata::seed_metadata(&label, rows, cols, &*z)
                            .await
                            .map_err(pyerr)?;
                    }
                    let matrix = dense_from_parts(rows, cols, data)?;
                    let md = z.metadata_path();
                    z.save_dense(&key, &matrix, &md).await.map_err(pyerr)?;
                    Ok(format!("{}/{id}", root.display()))
                })
            }
        }
    }

    #[pyo3(signature = (name))]
    fn load<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        match &self.backend {
            Backend::Lance(inner) => {
                let inner = inner.clone();
                future_into_py(py, async move {
                    let (rows, cols, flat) = inner.load_core(name).await?;
                    Python::attach(|py| flat_to_numpy(py, rows, cols, &flat))
                })
            }
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let key = Storage {
                    backend: self.backend.clone(),
                }
                .dataset_key(&name);
                future_into_py(py, async move {
                    let matrix = z.load_dense(&key).await.map_err(pyerr)?;
                    let (rows, cols, flat) = dense_to_flat(&matrix);
                    Python::attach(|py| flat_to_numpy(py, rows, cols, &flat))
                })
            }
        }
    }

    #[pyo3(signature = (name, array))]
    fn append<'py>(
        &self,
        py: Python<'py>,
        name: String,
        array: PyArrayLike2<'_, f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (rows, cols, data) = numpy_to_dense(&array)?;
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "append_vectors is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let z = storage.clone();
                let id = Storage {
                    backend: self.backend.clone(),
                }
                .make_dataset_id(&name);
                future_into_py(py, async move {
                    let matrix = dense_from_parts(rows, cols, data)?;
                    z.append_vectors(&id, &matrix).await.map_err(pyerr)
                })
            }
        }
    }

    #[pyo3(signature = (name, updates))]
    fn overwrite<'py>(
        &self,
        py: Python<'py>,
        name: String,
        updates: Vec<(usize, PyArrayLike1<'_, f64>)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.backend {
            Backend::Lance(_) => Err(StorageError::new_err(
                "overwrite_vectors is not supported on the lance backend; use the zarr backend",
            )),
            Backend::Zarr { storage, .. } => {
                let rows: Vec<RowUpdate> = updates
                    .into_iter()
                    .map(|(row_index, vector)| RowUpdate {
                        row_index,
                        vector: vector.as_array().iter().copied().collect(),
                    })
                    .collect();
                let z = storage.clone();
                let id = Storage {
                    backend: self.backend.clone(),
                }
                .make_dataset_id(&name);
                future_into_py(py, async move {
                    z.overwrite_vectors(&id, &rows).await.map_err(pyerr)
                })
            }
        }
    }

    fn __repr__(&self) -> String {
        "StorageAsync()".to_string()
    }
}

fn build_graph(
    src: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    weights: Option<PyArrayLike1<'_, f64>>,
    num_nodes: Option<u64>,
    node_id_width: Option<String>,
) -> PyResult<(Vec<GraphEdge>, GraphWriteOptions)> {
    let src = node_ids(src, "src")?;
    let dst = node_ids(dst, "dst")?;
    if src.len() != dst.len() {
        return Err(PyValueError::new_err(format!(
            "src length {} does not match dst length {}",
            src.len(),
            dst.len()
        )));
    }
    let weights: Option<Vec<f64>> = weights
        .as_ref()
        .map(|w| Ok::<Vec<f64>, PyErr>(w.as_array().iter().copied().collect()))
        .transpose()?;
    if let Some(w) = &weights
        && w.len() != src.len()
    {
        return Err(PyValueError::new_err(format!(
            "weights length {} does not match edge count {}",
            w.len(),
            src.len()
        )));
    }
    let edges: Vec<GraphEdge> = src
        .iter()
        .zip(dst.iter())
        .enumerate()
        .map(|(i, (&s, &d))| match &weights {
            Some(w) => GraphEdge::weighted(s, d, w[i]),
            None => GraphEdge::unweighted(s, d),
        })
        .collect();
    let width = match node_id_width.as_deref() {
        None => NodeIdWidth::U32,
        Some("u32") => NodeIdWidth::U32,
        Some("u64") => NodeIdWidth::U64,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown node_id_width '{other}': expected 'u32' or 'u64'"
            )));
        }
    };
    let options = GraphWriteOptions {
        node_id_width: width,
        num_nodes,
        ..Default::default()
    };
    Ok((edges, options))
}

fn summary_dict<'py>(py: Python<'py>, summary: &DatasetSummary) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("dataset_id", &summary.dataset_id)?;
    dict.set_item("root", &summary.root)?;
    dict.set_item("path", &summary.path)?;
    dict.set_item("shape", summary.shape.clone())?;
    dict.set_item("dtype", &summary.dtype)?;
    dict.set_item("chunks", summary.chunks.clone())?;
    dict.set_item(
        "fill_value",
        summary.fill_value.as_ref().map(|v| v.to_string()),
    )?;
    dict.set_item("kind", summary.kind.as_str())?;
    Ok(dict)
}

fn table_to_batch(obj: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    if let Ok(b) = obj.extract::<PyArrowType<RecordBatch>>() {
        return Ok(b.0);
    }
    if obj.hasattr("combine_chunks")? && obj.hasattr("to_batches")? {
        let combined = obj.call_method0("combine_chunks")?;
        let batches = combined.call_method0("to_batches")?;
        let list: Vec<PyArrowType<RecordBatch>> = batches.extract()?;
        let batch = list
            .into_iter()
            .next()
            .ok_or_else(|| PyValueError::new_err("table has no batches"))?;
        return Ok(batch.0);
    }
    Err(PyValueError::new_err(
        "expected a pyarrow Table or RecordBatch",
    ))
}

fn record_batch_to_py(py: Python<'_>, batch: RecordBatch) -> PyResult<Py<PyAny>> {
    let exported = PyArrowType(batch).into_pyobject(py)?;
    let list = PyList::new(py, [exported])?;
    let pa = py.import("pyarrow")?;
    let table = pa.getattr("Table")?.call_method1("from_batches", (list,))?;
    Ok(table.unbind())
}

fn graph_to_py(py: Python<'_>, graph: StoredGraph) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let src: Vec<u64> = graph.edges.iter().map(|e| e.src).collect();
    let dst: Vec<u64> = graph.edges.iter().map(|e| e.dst).collect();
    dict.set_item("src", PyArray1::from_slice(py, &src))?;
    dict.set_item("dst", PyArray1::from_slice(py, &dst))?;
    dict.set_item(
        "weight",
        if graph.weighted {
            let w: Vec<f64> = graph
                .edges
                .iter()
                .map(|e| e.weight.unwrap_or_default())
                .collect();
            PyArray1::from_slice(py, &w).into_any().unbind()
        } else {
            py.None()
        },
    )?;
    dict.set_item("num_nodes", graph.num_nodes)?;
    dict.set_item("weighted", graph.weighted)?;
    dict.set_item("node_id_width", graph.node_id_width.as_str())?;
    Ok(dict.into_any().unbind())
}

fn sparse_to_py(py: Python<'_>, matrix: &CsMat<f64>) -> PyResult<Py<PyAny>> {
    let mut indptr: Vec<i64> = vec![0];
    let mut indices: Vec<i64> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for row in matrix.outer_iterator() {
        for (col, &v) in row.iter() {
            indices.push(col as i64);
            data.push(v);
        }
        indptr.push(indices.len() as i64);
    }
    let indptr = PyArray1::from_slice(py, &indptr).into_any().unbind();
    let indices = PyArray1::from_slice(py, &indices).into_any().unbind();
    let data = PyArray1::from_slice(py, &data).into_any().unbind();
    Ok((indptr, indices, data, (matrix.rows(), matrix.cols()))
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

fn index_list(obj: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<usize>> {
    if let Ok(a) = obj.extract::<PyArrayLike1<'_, i64>>() {
        return a
            .as_array()
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyValueError::new_err(format!(
                        "{field} has a negative value ({v})"
                    )))
                } else {
                    Ok(v as usize)
                }
            })
            .collect();
    }
    if let Ok(a) = obj.extract::<PyArrayLike1<'_, i32>>() {
        return a
            .as_array()
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyValueError::new_err(format!(
                        "{field} has a negative value ({v})"
                    )))
                } else {
                    Ok(v as usize)
                }
            })
            .collect();
    }
    if let Ok(a) = obj.extract::<PyArrayLike1<'_, u32>>() {
        return Ok(a.as_array().iter().map(|&v| v as usize).collect());
    }
    if let Ok(a) = obj.extract::<PyArrayLike1<'_, u64>>() {
        return Ok(a.as_array().iter().map(|&v| v as usize).collect());
    }
    Err(PyValueError::new_err(format!(
        "{field} must be an integer numpy array"
    )))
}

fn csr_from_parts(
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: &PyArrayLike1<'_, f64>,
    shape: (usize, usize),
) -> PyResult<CsMat<f64>> {
    let values: Vec<f64> = data.as_array().iter().copied().collect();
    if indices.len() != values.len() {
        return Err(PyValueError::new_err(format!(
            "indices length {} does not match data length {}",
            indices.len(),
            values.len()
        )));
    }
    let (nrows, ncols) = shape;
    if indptr.len() != nrows + 1 {
        return Err(PyValueError::new_err(format!(
            "indptr length {} does not match nrows {} + 1",
            indptr.len(),
            nrows
        )));
    }
    let mut trimat = sprs::TriMat::with_capacity((nrows, ncols), values.len());
    for row in 0..nrows {
        let (start, end) = (indptr[row], indptr[row + 1]);
        if start > end || end > values.len() {
            return Err(PyValueError::new_err(
                "indptr is not monotonically increasing",
            ));
        }
        for k in start..end {
            let col = indices[k];
            if col >= ncols {
                return Err(PyValueError::new_err(format!(
                    "column index {col} out of bounds for {ncols} columns"
                )));
            }
            trimat.add_triplet(row, col, values[k]);
        }
    }
    Ok(trimat.to_csr())
}

/// Each directory is a separate storage.
/// If the same directory is passed, arrays are stored in the same storage.
#[pyfunction]
#[pyo3(name = "store_array", signature = (output_dir, backend=None, label=None))]
fn store_array(
    output_dir: String,
    backend: Option<String>,
    label: Option<String>,
) -> PyResult<StorageBuilder> {
    StorageBuilder::new(output_dir, backend, label, None, None, None)
}

/// Alias of `store_array` kept for the async API shown in the README.
#[pyfunction]
#[pyo3(name = "create_storage", signature = (output_dir, backend=None, label=None))]
fn create_storage(
    output_dir: String,
    backend: Option<String>,
    label: Option<String>,
) -> PyResult<StorageBuilder> {
    store_array(output_dir, backend, label)
}

/// Python module definition
#[pymodule]
fn genestore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init();

    m.add_class::<StorageBuilder>()?;
    m.add_class::<Storage>()?;
    m.add_class::<StorageAsync>()?;
    m.add("StorageError", m.py().get_type::<StorageError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(store_array, m)?)?;
    m.add_function(wrap_pyfunction!(create_storage, m)?)?;
    Ok(())
}
