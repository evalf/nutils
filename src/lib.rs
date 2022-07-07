pub mod finite_f64;
pub mod map;
pub mod simplex;
mod util;

use map::relative::RelativeTo;
use map::tesselation::Tesselation;
use map::transforms::Transforms;
use map::Map;
use numpy::{IntoPyArray, IxDyn, PyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadonlyArray2, PyArray2};
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

    fn apply_map_from_numpy<'py>(
        py: Python<'py>,
        map: &impl Map,
        index: usize,
        coords: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<(usize, &'py PyArrayDyn<f64>)> {
        if coords.ndim() == 0 {
            return Err(PyValueError::new_err(
                "the `coords` argument must have at least one dimension",
            ));
        }
        if coords.shape()[coords.ndim() - 1] != map.dim_in() {
            return Err(PyValueError::new_err(format!(
                "the last axis of the `coords` argument should have dimension {}",
                map.dim_in()
            )));
        }
        let mut result: Vec<f64> = coords
            .as_array()
            .rows()
            .into_iter()
            .flat_map(|row| {
                row.into_iter()
                    .cloned()
                    .chain(iter::repeat(0.0).take(map.delta_dim()))
            })
            .collect();
        let index = map.apply_inplace(index, &mut result, map.dim_out(), 0)?;
        let result = PyArray::from_vec(py, result);
        let shape: Vec<usize> = coords
            .shape()
            .iter()
            .take(coords.ndim() - 1)
            .cloned()
            .chain(iter::once(map.dim_out()))
            .collect();
        let result = result.reshape(&shape[..])?;
        Ok((index, result))
    }

    #[pyclass(name = "Tesselation", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyTesselation(Tesselation);

    #[pymethods]
    impl PyTesselation {
        #[staticmethod]
        pub fn new_identity(shapes: Vec<PySimplex>, len: usize) -> Self {
            let shapes = shapes.iter().map(|shape| shape.into()).collect();
            Tesselation::identity(shapes, len).into()
        }
        pub fn __repr__(&self) -> String {
            format!("{:?}", self.0)
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
        pub fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
            Ok(Self(self.0.take(&indices)?))
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
            apply_map_from_numpy(py, &self.0, index, coords)
        }
        //pub fn unapply_indices(&self, indices: Vec<usize>) -> PyResult<Vec<usize>> {
        //    self.0
        //        .unapply_indices(&indices)
        //        .map(|mut indices| { indices.sort(); indices })
        //        .ok_or(PyValueError::new_err("index out of range"))
        //}
        pub fn relative_to(&self, target: &Self) -> PyResult<PyMap> {
            self.0
                .relative_to(&target.0)
                .map(|rel| PyMap(rel))
                .ok_or(PyValueError::new_err("cannot make relative"))
        }
    }

    impl From<Tesselation> for PyTesselation {
        fn from(tesselation: Tesselation) -> PyTesselation {
            PyTesselation(tesselation)
        }
    }

    m.add_class::<PyTesselation>()?;

    #[pyclass(name = "Map", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyMap(<Tesselation as RelativeTo<Tesselation>>::Output);

    #[pymethods]
    impl PyMap {
        pub fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        pub fn len_out(&self) -> usize {
            self.0.len_out()
        }
        pub fn len_in(&self) -> usize {
            self.0.len_in()
        }
        pub fn dim_out(&self) -> usize {
            self.0.dim_out()
        }
        pub fn dim_in(&self) -> usize {
            self.0.dim_in()
        }
        pub fn delta_dim(&self) -> usize {
            self.0.delta_dim()
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
            apply_map_from_numpy(py, &self.0, index, coords)
        }
        pub fn unapply_indices(&self, indices: Vec<usize>) -> PyResult<Vec<usize>> {
            self.0
                .unapply_indices(&indices)
                .map(|mut indices| {
                    indices.sort();
                    indices
                })
                .ok_or(PyValueError::new_err("index out of range"))
        }
    }

    m.add_class::<PyMap>()?;

    #[pyclass(name = "Transforms", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyTransforms(Transforms);

    #[pymethods]
    impl PyTransforms {
        #[staticmethod]
        pub fn new_identity(dim: usize, len: usize) -> Self {
            Transforms::identity(dim, len).into()
        }
        pub fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        pub fn __len__(&self) -> usize {
            self.0.len_in()
        }
        #[getter]
        pub fn fromlen(&self) -> usize {
            self.0.len_out()
        }
        #[getter]
        pub fn tolen(&self) -> usize {
            self.0.len_out()
        }
        #[getter]
        pub fn fromdims(&self) -> usize {
            self.0.dim_in()
        }
        #[getter]
        pub fn todims(&self) -> usize {
            self.0.dim_out()
        }
        pub fn __mul__(&self, rhs: &PyTransforms) -> Self {
            Self(&self.0 * &rhs.0)
        }
        pub fn concat(&self, other: &PyTransforms) -> PyResult<Self> {
            Ok(Self(self.0.concat(&other.0)?))
        }
        pub fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
            Ok(Self(self.0.take(&indices)?))
        }
        pub fn children(&self, simplex: &PySimplex, offset: usize) -> PyResult<Self> {
            Ok(Self(self.0.children(simplex.into(), offset)?))
        }
        pub fn edges(&self, simplex: &PySimplex, offset: usize) -> PyResult<Self> {
            Ok(Self(self.0.edges(simplex.into(), offset)?))
        }
        pub fn uniform_points(
            &self,
            points: PyReadonlyArray2<f64>,
            offset: usize,
        ) -> PyResult<Self> {
            let point_dim = points.shape()[1];
            let points: Vec<f64> = points.as_array().iter().cloned().collect();
            Ok(Self(self.0.uniform_points(points, point_dim, offset)?))
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
            apply_map_from_numpy(py, &self.0, index, coords)
        }
        //pub fn unapply_indices(&self, indices: Vec<usize>) -> PyResult<Vec<usize>> {
        //    self.0
        //        .unapply_indices(&indices)
        //        .map(|mut indices| { indices.sort(); indices })
        //        .ok_or(PyValueError::new_err("index out of range"))
        //}
        pub fn relative_to(&self, target: &Self) -> PyResult<PyRelativeTransforms> {
            self.0
                .relative_to(&target.0)
                .map(|rel| PyRelativeTransforms(rel))
                .ok_or(PyValueError::new_err("cannot make relative"))
        }
        pub fn basis<'py>(&self, py: Python<'py>, index: usize) -> PyResult<&'py PyArray2<f64>> {
            if index >= self.0.len_in() {
                return Err(PyIndexError::new_err("index out of range"));
            }
            let mut basis: Vec<f64> = iter::repeat(0.0).take(self.0.dim_out() * self.0.dim_out()).collect();
            for i in 0..self.0.dim_in() {
                basis[i * self.0.dim_out() + i] = 1.0;
            }
            let mut dim_in = self.0.dim_in();
            self.0.update_basis(index, &mut basis[..], self.0.dim_out(), &mut dim_in, 0);
            PyArray::from_vec(py, basis).reshape([self.0.dim_out(), self.0.dim_out()])
        }
        #[getter]
        pub fn is_identity(&self) -> bool {
            self.0.is_identity()
        }
        #[getter]
        pub fn is_index_map(&self) -> bool {
            self.0.is_index_map()
        }
    }

    impl From<Transforms> for PyTransforms {
        fn from(transforms: Transforms) -> PyTransforms {
            PyTransforms(transforms)
        }
    }

    m.add_class::<PyTransforms>()?;

    #[pyclass(name = "RelativeTransforms", module = "nutils._rust")]
    #[derive(Debug, Clone)]
    struct PyRelativeTransforms(<Transforms as RelativeTo<Transforms>>::Output);

    #[pymethods]
    impl PyRelativeTransforms {
        pub fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }
        pub fn __len__(&self) -> usize {
            self.0.len_in()
        }
        #[getter]
        pub fn fromlen(&self) -> usize {
            self.0.len_out()
        }
        #[getter]
        pub fn tolen(&self) -> usize {
            self.0.len_out()
        }
        #[getter]
        pub fn fromdims(&self) -> usize {
            self.0.dim_in()
        }
        #[getter]
        pub fn todims(&self) -> usize {
            self.0.dim_out()
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
            apply_map_from_numpy(py, &self.0, index, coords)
        }
        pub fn unapply_indices(&self, indices: Vec<usize>) -> PyResult<Vec<usize>> {
            self.0
                .unapply_indices(&indices)
                .map(|mut indices| {
                    indices.sort();
                    indices
                })
                .ok_or(PyValueError::new_err("index out of range"))
        }
        pub fn basis<'py>(&self, py: Python<'py>, index: usize) -> PyResult<&'py PyArray2<f64>> {
            if index >= self.0.len_in() {
                return Err(PyIndexError::new_err("index out of range"));
            }
            let mut basis: Vec<f64> = iter::repeat(0.0).take(self.0.dim_out() * self.0.dim_out()).collect();
            for i in 0..self.0.dim_in() {
                basis[i * self.0.dim_out() + i] = 1.0;
            }
            let mut dim_in = self.0.dim_in();
            self.0.update_basis(index, &mut basis[..], self.0.dim_out(), &mut dim_in, 0);
            PyArray::from_vec(py, basis).reshape([self.0.dim_out(), self.0.dim_out()])
        }
        #[getter]
        pub fn is_identity(&self) -> bool {
            self.0.is_identity()
        }
        #[getter]
        pub fn is_index_map(&self) -> bool {
            self.0.is_index_map()
        }
    }

    m.add_class::<PyRelativeTransforms>()?;

    Ok(())
}
