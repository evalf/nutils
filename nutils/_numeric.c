#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"
#include "float.h"

#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); Py_Exit( 1 ); }

typedef struct { int countdown, resetcounter, stride0, stride1; } DataStepper;

static PyObject *numeric_contract( PyObject *self, PyObject *args, PyObject *kwargs ) {
  // Contracts two array-like objects over specified axis int/axes tuple. Fully
  // equivalent with pointwise multiplication followed by summation:
  // (A*B).sum(n) == contract(A,B,n).

  PyObject *A = NULL, *B = NULL;
  int ncontract = -1;
  char *keywords[] = { "a", "b", "ncontract", NULL };
  int nd;
  DataStepper axes[NPY_MAXDIMS];
  int i, n;
  PyObject *C;
  double *ptrA, *ptrB, *ptrC;
  if ( ! PyArg_ParseTupleAndKeywords( args, kwargs, "OOi", keywords, &A, &B, &ncontract ) ) {
    return NULL;
  }
  if ( ! PyArray_Check(A) || ! PyArray_Check(B) ) {
    PyErr_Format( PyExc_TypeError, "expected numpy arrays" );
    return NULL;
  }
  nd = PyArray_NDIM(A);
  if ( nd != PyArray_NDIM(B) ) {
    PyErr_Format( PyExc_TypeError, "dimensions do not match" );
    return NULL;
  }
  if ( ncontract < 0 || ncontract > nd ) {
    PyErr_Format( PyExc_TypeError, "invalid ncontract" );
    return NULL;
  }
  for ( i = 0; i < nd; i++ ) {
    n = PyArray_DIM(A,i);
    if ( n == 0 ) {
      PyErr_Format( PyExc_TypeError, "contraction over zero-axis" );
      return NULL;
    }
    if ( n != PyArray_DIM(B,i) ) {
      PyErr_Format( PyExc_TypeError, "shapes do not match" );
      return NULL;
    }
    axes[i].countdown = axes[i].resetcounter = n-1;
    axes[i].stride0 = PyArray_STRIDE(A,i) / sizeof(double);
    axes[i].stride1 = PyArray_STRIDE(B,i) / sizeof(double);
  }
  C = PyArray_EMPTY( nd-ncontract, PyArray_DIMS(A), NPY_DOUBLE, 0 );
  if ( C == NULL ) {
    return NULL;
  }
  if ( PyArray_SIZE( C ) == 0 ) {
    return C;
  }
  ptrA = PyArray_DATA( A );
  ptrB = PyArray_DATA( B );
  ptrC = PyArray_DATA( C );
  (*ptrC) = (*ptrA) * (*ptrB);
  for ( i = nd-1; i >= 0; i-- ) {
    if ( axes[i].countdown == 0 ) {
      ptrA -= axes[i].stride0 * axes[i].resetcounter;
      ptrB -= axes[i].stride1 * axes[i].resetcounter;
      axes[i].countdown = axes[i].resetcounter;
    }
    else {
      axes[i].countdown--;
      ptrA += axes[i].stride0;
      ptrB += axes[i].stride1;
      if ( i >= nd-ncontract ) {
        (*ptrC) += (*ptrA) * (*ptrB);
      }
      else {
        ptrC++;
        (*ptrC) = (*ptrA) * (*ptrB);
      }
      i = nd;
    }
  }
  return C;
}

static PyMethodDef module_methods[] = {
  // List of python-exposed methods.

  { "contract", (PyCFunction)numeric_contract, METH_KEYWORDS, "contract" },
  { NULL },
};

static PyObject *numeric_module;
PyMODINIT_FUNC init_numeric( void ) {
  // Initializes module and numpy components.

  numeric_module = Py_InitModule( "_numeric", module_methods );
  import_array();
}

// vim:shiftwidth=2:foldmethod=syntax
