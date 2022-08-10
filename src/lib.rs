#![cfg_attr(feature = "bench", feature(test))]

mod poly;

use pyo3::prelude::*;

fn add_submodule<'py>(
    py: Python<'py>,
    sys_modules: &PyAny,
    parent: &PyModule,
    name: &str,
) -> PyResult<&'py PyModule> {
    let submodule = PyModule::new(py, name)?;
    parent.add_submodule(submodule)?;
    sys_modules.set_item(
        format!("{}.{}", parent.getattr("__name__")?, name),
        submodule,
    )?;
    Ok(submodule)
}

#[pymodule]
fn _rust(py: Python, m: &PyModule) -> PyResult<()> {
    let sys_modules = py.import("sys")?.getattr("modules")?;
    let m_poly = add_submodule(py, sys_modules, m, "poly")?;

    m_poly.add_function(wrap_pyfunction!(poly::degree, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::ncoeffs, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::eval, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::grad, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::deriv, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::mul, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::outer_mul, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::transform_matrix, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::convert_old, m_poly)?)?;
    m_poly.add_function(wrap_pyfunction!(poly::change_degree, m_poly)?)?;

    Ok(())
}
