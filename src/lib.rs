use pyo3::prelude::*;

#[pymodule]
#[allow(non_snake_case)]
fn _rust(_py: Python, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
