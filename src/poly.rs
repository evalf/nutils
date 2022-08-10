use ndarray::{ArrayD, Dimension};
use numpy::{PyArray, PyArrayDyn, PyReadonlyArrayDyn, ToPyArray};
use nutils_poly::{Poly, PolySequence, Power, Powers, Variables};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::iter;

macro_rules! shape {
    ($($part:expr),*) => {{
        let mut shape: Vec<usize> = Vec::new();
        $(shape.extend($part.into_iter().map(|item| item.to_owned()));)*
        shape
    }};
    ($($part:expr,)*) => { shape![$($part),*] };
}

fn broadcast_lengths(mut lengths: impl Iterator<Item = usize>) -> PyResult<usize> {
    lengths.try_fold(1, |m, n| {
        if m == 1 || m == n {
            Ok(n)
        } else if n == 1 {
            Ok(m)
        } else {
            Err(PyValueError::new_err("incompatible lengths"))
        }
    })
}

fn broadcast_shapes(shapes: &[&[usize]]) -> PyResult<Vec<usize>> {
    let n = shapes
        .iter()
        .map(|shape| shape.len())
        .max()
        .ok_or_else(|| PyValueError::new_err("cannot broadcast zero shapes"))?;
    (0..n)
        .map(|i| {
            broadcast_lengths(
                shapes
                    .iter()
                    .map(|shape| i.checked_sub(n - shape.len()).map_or(1, |j| shape[j])),
            )
        })
        .collect()
}

/// Returns the degree of a polynomial given the number of coefficients and the dimension.
#[pyfunction]
#[pyo3(text_signature = "(ncoeffs, dim)")]
pub fn degree(ncoeffs: usize, dim: usize) -> PyResult<Power> {
    let degree = (0..)
        .take_while(|degree| nutils_poly::ncoeffs(dim, *degree) < ncoeffs)
        .count() as Power;
    if ncoeffs == nutils_poly::ncoeffs(dim, degree) {
        Ok(degree)
    } else {
        Err(PyValueError::new_err("invalid number of coefficients"))
    }
}

/// Returns the length of the coefficient vector for a polynomial of given degree and dimension.
#[pyfunction]
#[pyo3(text_signature = "(dim, degree")]
pub fn ncoeffs(dim: usize, degree: Power) -> usize {
    nutils_poly::ncoeffs(dim, degree)
}

/// Evaluates a polynomial at the given coordinates.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, coords)")]
pub fn eval<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    coords: PyReadonlyArrayDyn<f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs = coeffs.as_array();
    let coords = coords.as_array();
    if coeffs.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    if coords.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the `coords` argument must have at least one axis",
        ));
    }
    let dim = *coords.shape().last().unwrap();
    let ncoeffs = *coeffs.shape().last().unwrap();
    let degree = degree(ncoeffs, dim)?;
    let shape = shape![
        &coords.shape()[..coords.ndim() - 1],
        &coeffs.shape()[..coeffs.ndim() - 1],
    ];
    let mut result = Vec::with_capacity(shape.iter().copied().product());
    //if coeffs.strides().last() == Some(&1) && coords.strides().last() == Some(&1) {
    //    for coord in coords.rows() {
    //        for coeffs in coeffs.rows() {
    //            let poly = PolySequence::new(coeffs, 0..dim, degree).unwrap();
    //            result.push(poly.eval(coord.as_slice().unwrap()))
    //        }
    //    }
    //} else {
    {
        for coord in coords.rows() {
            for coeffs in coeffs.rows() {
                result.push(PolySequence::new(coeffs, 0..dim, degree).unwrap().eval(&coord))
            }
        }
    }
    PyArray::from_vec(py, result).reshape(shape)
}

/// Returns the gradient of a polynomial with the given dimension.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, dim)")]
pub fn grad<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    dim: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs = coeffs.as_array();
    if coeffs.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    let degree = degree(*coeffs.shape().last().unwrap(), dim)?;
    let grad_degree = degree.saturating_sub(1);
    let shape = shape![
        &coeffs.shape()[..coeffs.ndim() - 1],
        [dim, nutils_poly::ncoeffs(dim, grad_degree)],
    ];
    if degree == 0 {
        Ok(PyArray::zeros(py, shape, false))
    } else {
        let mut result = Vec::with_capacity(shape.iter().copied().product::<usize>() * dim);
        for coeffs in coeffs.rows() {
            let src = PolySequence::new(coeffs, 0..dim, degree).unwrap();
            for ideriv in 0..dim {
                result.extend(
                    src.by_ref()
                        .partial_deriv(ideriv.try_into().unwrap())
                        .coeffs_iter(),
                )
            }
        }
        PyArray::from_vec(py, result).reshape(shape)
    }
}

/// Returns the derivative of a polynomial.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, dim)")]
pub fn deriv<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    dim: usize,
    ideriv: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs = coeffs.as_array();
    if coeffs.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    if ideriv >= dim {
        return Err(PyValueError::new_err(
            "the derivative index is out of range",
        ));
    }
    let degree = degree(*coeffs.shape().last().unwrap(), dim)?;
    let deriv_degree = degree.saturating_sub(1);
    let shape = shape![
        &coeffs.shape()[..coeffs.ndim() - 1],
        [nutils_poly::ncoeffs(dim, deriv_degree)],
    ];
    if degree == 0 {
        Ok(PyArray::zeros(py, shape, false))
    } else {
        let mut result = Vec::with_capacity(shape.iter().copied().product());
        for coeffs in coeffs.rows() {
            let src = PolySequence::new(coeffs, 0..dim, degree).unwrap();
            result.extend(src.partial_deriv(ideriv.try_into().unwrap()).coeffs_iter());
        }
        PyArray::from_vec(py, result).reshape(shape)
    }
}

/// Returns the product of two polynomials.
#[pyfunction]
#[pyo3(text_signature = "(coeffs1, coeffs2, dim)")]
pub fn mul<'py>(
    py: Python<'py>,
    coeffs1: PyReadonlyArrayDyn<f64>,
    coeffs2: PyReadonlyArrayDyn<f64>,
    dim: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs1 = coeffs1.as_array();
    let coeffs2 = coeffs2.as_array();
    if coeffs1.ndim() == 0 || coeffs2.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    let ncoeffs1 = *coeffs1.shape().last().unwrap();
    let ncoeffs2 = *coeffs2.shape().last().unwrap();
    let degree1 = degree(ncoeffs1, dim)?;
    let degree2 = degree(ncoeffs2, dim)?;
    let common_shape = broadcast_shapes(&[
        &coeffs1.shape()[..coeffs1.ndim() - 1],
        &coeffs2.shape()[..coeffs2.ndim() - 1],
    ])?;
    let coeffs1 = coeffs1
        .broadcast(shape![&common_shape, [ncoeffs1]])
        .unwrap();
    let coeffs2 = coeffs2
        .broadcast(shape![&common_shape, [ncoeffs2]])
        .unwrap();
    let result_shape = shape![
        &common_shape,
        [nutils_poly::ncoeffs(dim, degree1 + degree2)]
    ];
    let mut coeffsr = ArrayD::zeros(result_shape);
    //let mut result = Vec::with_capacity(result_shape.iter().copied().product());
    //if coeffs1.strides().last() == Some(&1) && coeffs2.strides().last() == Some(&1) {
    //    for (coeffs1, coeffs2) in iter::zip(coeffs1.rows(), coeffs2.rows()) {
    //        let poly1 = PolySequence::new(coeffs1.as_slice().unwrap(), 0..dim, degree1).unwrap();
    //        let poly2 = PolySequence::new(coeffs2.as_slice().unwrap(), 0..dim, degree2).unwrap();
    //        result.extend(poly1.mul(&poly2).collect::<Vec<_>>().coeffs_iter());
    //    }
    //} else {
    {
        for (coeffsr, (coeffs1, coeffs2)) in iter::zip(coeffsr.rows_mut(), iter::zip(coeffs1.rows(), coeffs2.rows())) {
            let poly1 = PolySequence::new(coeffs1, 0..dim, degree1).unwrap();
            let poly2 = PolySequence::new(coeffs2, 0..dim, degree2).unwrap();
            let mut polyr = PolySequence::new(coeffsr, 0..dim, degree1 + degree2).unwrap();
            poly1.mul(&poly2).add_to(&mut polyr).unwrap();
        }
    }
    Ok(coeffsr.to_pyarray(py))
    //PyArray::from_vec(py, result).reshape(result_shape)
}

/// Returns the outer product of two polynomials.
#[pyfunction]
#[pyo3(text_signature = "(coeffs1, coeffs2, dim)")]
pub fn outer_mul<'py>(
    py: Python<'py>,
    coeffs1: PyReadonlyArrayDyn<f64>,
    coeffs2: PyReadonlyArrayDyn<f64>,
    dim1: usize,
    dim2: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs1 = coeffs1.as_array();
    let coeffs2 = coeffs2.as_array();
    if coeffs1.ndim() == 0 || coeffs2.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    let ncoeffs1 = *coeffs1.shape().last().unwrap();
    let ncoeffs2 = *coeffs2.shape().last().unwrap();
    let degree1 = degree(ncoeffs1, dim1)?;
    let degree2 = degree(ncoeffs2, dim2)?;
    let common_shape = broadcast_shapes(&[
        &coeffs1.shape()[..coeffs1.ndim() - 1],
        &coeffs2.shape()[..coeffs2.ndim() - 1],
    ])?;
    let ncoeffsr = nutils_poly::ncoeffs(dim1 + dim2, degree1 + degree2);
    let coeffs1 = coeffs1
        .broadcast(shape![&common_shape, [ncoeffs1]])
        .unwrap();
    let coeffs2 = coeffs2
        .broadcast(shape![&common_shape, [ncoeffs2]])
        .unwrap();
    let result_shape = shape![&common_shape, [ncoeffsr]];
    let mut coeffsr = ArrayD::zeros(result_shape);
    //if coeffs1.strides().last() == Some(&1) && coeffs2.strides().last() == Some(&1) {
    //    for (coeffs1, coeffs2) in iter::zip(coeffs1.rows(), coeffs2.rows()) {
    //        let poly1 = PolySequence::new(coeffs1.as_slice().unwrap(), 0..dim1, degree1).unwrap();
    //        let poly2 =
    //            PolySequence::new(coeffs2.as_slice().unwrap(), dim1..dim1 + dim2, degree2).unwrap();
    //        poly1
    //            .mul(&poly2)
    //            .add_to(&mut result_polys.next().unwrap())
    //            .unwrap();
    //    }
    //} else {
    {
        for (coeffsr, (coeffs1, coeffs2)) in iter::zip(coeffsr.rows_mut(), iter::zip(coeffs1.rows(), coeffs2.rows())) {
            let poly1 = PolySequence::new(coeffs1, 0..dim1, degree1).unwrap();
            let poly2 = PolySequence::new(coeffs2, dim1..dim1 + dim2, degree2).unwrap();
            let mut polyr = PolySequence::new(coeffsr, 0..dim1 + dim2, degree1 + degree2).unwrap();
            poly1.mul(&poly2).add_to(&mut polyr).unwrap();
        }
    }
    Ok(coeffsr.to_pyarray(py))
}

/// Returns a transformation matrix.
///
/// The matrix is such that the following two expressions are equivalent:
///
///     eval(coeffs, eval(transform_coeffs, coord, from_dim), poly_dim)
///     eval(matvec(matrix, coeffs), coord, from_dim)
///
/// where `matrix` is the result of
///
///     transform_matrix(transform_coeffs, transform_degree, from_dim, poly_degree, poly_dim)
#[pyfunction]
#[pyo3(text_signature = "(transform_coeffs, from_dim, poly_degree, poly_dim)")]
pub fn transform_matrix<'py>(
    py: Python<'py>,
    transform_coeffs: PyReadonlyArrayDyn<f64>,
    from_dim: usize,
    poly_degree: Power,
    poly_dim: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let transform_coeffs = transform_coeffs.as_array();
    if transform_coeffs.ndim() != 2 {
        return Err(PyValueError::new_err(
            "the transformation coefficients must have two axes",
        ));
    }
    if transform_coeffs.shape()[0] != poly_dim {
        return Err(PyValueError::new_err(
            "invalid shapeo of the transformation coefficients",
        ));
    }
    let transform_degree = degree(transform_coeffs.shape()[1], from_dim)?;
    let result = nutils_poly::transform_matrix(
        transform_coeffs.as_standard_layout().as_slice().unwrap(),
        transform_degree,
        from_dim,
        poly_degree,
        poly_dim,
    );
    let shape = [
        nutils_poly::ncoeffs(from_dim, transform_degree * poly_degree),
        nutils_poly::ncoeffs(poly_dim, poly_degree),
    ];
    // TODO: transpose
    PyArray::from_vec(py, result).reshape(&shape[..])
}

/// Convert old style coefficients.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, dim)")]
pub fn convert_old<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    dim: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs = coeffs.as_array();
    if coeffs.ndim() < dim {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis per dimension",
        ));
    }
    if dim < 1 {
        Ok(PyArray::from_owned_array(py, coeffs.to_owned()))
    } else {
        let n = *coeffs.shape().last().unwrap();
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "the last {dim} axes must have nonzero length"
            )));
        }
        if coeffs.shape()[coeffs.ndim() - dim..]
            .iter()
            .any(|m| *m != n)
        {
            return Err(PyValueError::new_err(format!(
                "the last {dim} axes of the coefficients must have the same lengths"
            )));
        }
        let degree = n as Power - 1;
        let ncoeffs = nutils_poly::ncoeffs(dim, degree);
        let vars = Variables::try_from(0..dim).unwrap();
        let n = coeffs.ndim() - dim;
        let shape = shape![&coeffs.shape()[..n], [ncoeffs]];
        let mut result: ArrayD<f64> = ArrayD::zeros(&shape[..]);
        let mut new_index: Vec<usize> = iter::repeat(0).take(shape.len()).collect();
        for (index, coeff) in coeffs.indexed_iter() {
            let mut powers = Powers::zeros();
            for (v, p) in iter::zip(vars.iter(), &index.slice()[n..]) {
                powers[v] = *p as Power;
            }
            if let Some(i) = powers.to_index(vars, degree) {
                new_index[..n].copy_from_slice(&index.slice()[..n]);
                new_index[n] = i;
                result[&new_index[..]] = *coeff;
            }
        }
        Ok(PyArray::from_owned_array(py, result))
    }
}

/// Return coefficients for the given degree.
#[pyfunction]
#[pyo3(text_signature = "(coeffs, new_degree, dim)")]
pub fn change_degree<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArrayDyn<f64>,
    new_degree: Power,
    dim: usize,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let coeffs = coeffs.as_array();
    if coeffs.ndim() == 0 {
        return Err(PyValueError::new_err(
            "the coefficients must have at least one axis",
        ));
    }
    let degree = degree(*coeffs.shape().last().unwrap(), dim)?;
    match new_degree.cmp(&degree) {
        Ordering::Less => Err(PyValueError::new_err(
            "the new degree is lower than the old degree",
        )),
        Ordering::Equal => Ok(PyArray::from_owned_array(py, coeffs.to_owned())),
        Ordering::Greater => {
            let new_ncoeffs = nutils_poly::ncoeffs(dim, new_degree);
            let n = coeffs.ndim() - 1;
            let new_shape = shape![&coeffs.shape()[..n], [new_ncoeffs]];
            let mut new_coeffs: ArrayD<f64> = ArrayD::zeros(&new_shape[..]);
            for (new_coeffs, coeffs) in iter::zip(new_coeffs.rows_mut(), coeffs.rows()) {
                let src = PolySequence::new(coeffs, 0..dim, degree).unwrap();
                let mut dst =
                    PolySequence::new(new_coeffs, 0..dim, new_degree).unwrap();
                src.add_to(&mut dst).unwrap();
            }
            Ok(PyArray::from_owned_array(py, new_coeffs))
        }
    }
}
