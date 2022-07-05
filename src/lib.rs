pub mod finite_f64;
pub mod map;
pub mod simplex;
mod util;

use map::tesselation::Tesselation;
use map::Map;
use numpy::{IntoPyArray, IxDyn, PyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use simplex::Simplex;
use std::iter;

impl From<map::Error> for PyErr {
    fn from(err: map::Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pymodule]
#[allow(non_snake_case)]
fn _rust(py: Python, m: &PyModule) -> PyResult<()> {
    let sys_modules = py.import("sys")?.getattr("modules")?;

    #[pyclass(name = "Simplex", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PySimplex(Simplex);

    #[pymethods]
    impl PySimplex {
        #[classattr]
        pub fn line() -> Self {
            Simplex::Line.into()
        }
        #[classattr]
        pub fn triangle() -> Self {
            Simplex::Triangle.into()
        }
        #[getter]
        pub fn dim(&self) -> usize {
            self.0.dim()
        }
        #[getter]
        pub fn edge_dim(&self) -> usize {
            self.0.edge_dim()
        }
        #[getter]
        pub fn edge_simplex(&self) -> Option<Self> {
            self.0.edge_simplex().map(|simplex| simplex.into())
        }
        #[getter]
        pub fn nchildren(&self) -> usize {
            self.0.nchildren()
        }
        #[getter]
        pub fn nedges(&self) -> usize {
            self.0.nedges()
        }
    }

    impl From<Simplex> for PySimplex {
        fn from(simplex: Simplex) -> PySimplex {
            PySimplex(simplex)
        }
    }

    impl From<PySimplex> for Simplex {
        fn from(pysimplex: PySimplex) -> Simplex {
            pysimplex.0
        }
    }

    impl From<&PySimplex> for Simplex {
        fn from(pysimplex: &PySimplex) -> Simplex {
            pysimplex.0
        }
    }

    m.add_class::<PySimplex>()?;

    #[pyclass(name = "Tesselation", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyTesselation(Tesselation);

    #[pymethods]
    impl PyTesselation {
        #[staticmethod]
        pub fn identity(shapes: Vec<PySimplex>, len: usize) -> Self {
            let shapes = shapes.iter().map(|shape| shape.into()).collect();
            Tesselation::identity(shapes, len).into()
        }
        pub fn __len__(&self) -> usize {
            self.0.len()
        }
        #[getter]
        pub fn dim(&self) -> usize {
            self.0.len()
        }
        pub fn __mul__(&self, rhs: &PyTesselation) -> Self {
            Self(&self.0 * &rhs.0)
        }
        pub fn concat(&self, other: &PyTesselation) -> PyResult<Self> {
            Ok(Self(self.0.concat(&other.0)?))
        }
        pub fn take(&self, indices: Vec<usize>) -> Self {
            Self(self.0.take(&indices))
        }
        #[getter]
        pub fn children(&self) -> Self {
            Self(self.0.children())
        }
        #[getter]
        pub fn edges(&self) -> PyResult<Self> {
            Ok(Self(self.0.edges()?))
        }
        #[getter]
        pub fn centroids(&self) -> Self {
            Self(self.0.centroids())
        }
        #[getter]
        pub fn vertices(&self) -> Self {
            Self(self.0.vertices())
        }
        pub fn apply_index(&self, index: usize) -> PyResult<usize> {
            self.0
                .apply_index(index)
                .ok_or(PyIndexError::new_err("index out of range"))
        }
        pub fn apply<'py>(
            &self,
            py: Python<'py>,
            index: usize,
            coords: PyReadonlyArrayDyn<f64>,
        ) -> PyResult<(usize, &'py PyArrayDyn<f64>)> {
            if coords.ndim() == 0 {
                return Err(PyValueError::new_err(
                    "the `coords` argument must have at least one dimension",
                ));
            }
            if coords.shape()[coords.ndim() - 1] != self.0.dim() {
                return Err(PyValueError::new_err(format!(
                    "the last axis of the `coords` argument should have dimension {}",
                    self.0.dim()
                )));
            }
            let mut result: Vec<f64> = coords
                .as_array()
                .rows()
                .into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .cloned()
                        .chain(iter::repeat(0.0).take(self.0.delta_dim()))
                })
                .collect();
            let index = self.0.apply_inplace(index, &mut result, self.0.dim_out())?;
            let result = PyArray::from_vec(py, result);
            let shape: Vec<usize> = coords
                .shape()
                .iter()
                .take(coords.ndim() - 1)
                .cloned()
                .chain(iter::once(self.0.dim_out()))
                .collect();
            let result = result.reshape(&shape[..])?;
            Ok((index, result))
        }
    }

    impl From<Tesselation> for PyTesselation {
        fn from(tesselation: Tesselation) -> PyTesselation {
            PyTesselation(tesselation)
        }
    }

    m.add_class::<PyTesselation>()?;

    Ok(())
}
