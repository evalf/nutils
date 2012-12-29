#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"
#include "float.h"

#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); Py_Exit( 1 ); }

typedef struct { int countdown, resetcounter, stride0, stride1; } DataStepper;

static inline PyObject *new_dims_and_strides( PyObject *array, int nd, npy_intp *dims, npy_intp *strides, int flags ) {
  // Creates a new view of existing data, with specified number of axes, shape,
  // and strides. The original array becomes the base of the old array without
  // (!) increasing its reference count. On failure decreases the original
  // matrix' reference count and returns NULL.

  PyArray_Descr *descr = PyArray_DESCR(array);
  PyObject *newarray = PyArray_NewFromDescr( &PyArray_Type, descr, nd, dims, strides, PyArray_DATA(array), flags, NULL );
  if ( newarray == NULL ) {
    Py_DECREF( array );
  }
  else {
    Py_INCREF( descr );
    ((PyArrayObject *)newarray)->base = array;
  }
  return newarray;
}

static PyObject *numeric_addsorted( PyObject *self, PyObject *args, PyObject *kwargs ) {
  // Adds numbers into sorted array of non-negative integers. Sorting is not
  // checked. Both arguments must be castable to int arrays. First argument
  // data (not size) may be modified. Maintains a base array of power of two
  // length to minimize reallocations.

  PyObject *array=NULL, *entries=NULL;
  int inplace = 0;
  char *keywords[] = { "array", "entries", "inplace", NULL };
  if ( ! PyArg_ParseTupleAndKeywords( args, kwargs, "O&O&|i", keywords, PyArray_Converter, &array, PyArray_Converter, &entries, &inplace ) ) {
    Py_XDECREF( array );
    Py_XDECREF( entries );
    return NULL;
  }
  if ( PyArray_NDIM(array) != 1 ) {
    PyErr_Format( PyExc_TypeError, "arguments should by 1-dimensional int-arrays" );
    Py_DECREF( array );
    Py_DECREF( entries );
    return NULL;
  }
  npy_intp size = PyArray_SIZE(array);
  npy_intp N = 1;
  while ( N <= size + PyArray_SIZE(entries) ) {
    N <<= 1;
  }
  int i;
  PyObject *base = PyArray_BASE(array);
  if ( !inplace || base == NULL
                || !PyArray_Check(base)
                || !PyArray_ISCONTIGUOUS(base)
                || PyArray_NDIM(base) != 1
                || PyArray_SIZE(base) != N
                || PyArray_DATA(array) != PyArray_DATA(base) ) {
    base = PyArray_SimpleNew( 1, &N, NPY_INT );
    if PyArray_ISCONTIGUOUS( array ) {
      memcpy( PyArray_DATA(base), PyArray_DATA(array), size * sizeof(npy_int) );
    }
    else {
      npy_int* dst = PyArray_DATA(base);
      for ( i = 0; i < size; i++ ) {
        dst[i] = *(npy_int*)PyArray_GETPTR1(array,i);
      }
    }
  }
  else {
    Py_INCREF( base );
  }
  Py_DECREF( array );
  npy_int *data = PyArray_DATA( base ); // contigous length N
  PyObject *iter = PyArray_IterNew( entries );
  ASSERT( iter );
  while ( PyArray_ITER_NOTDONE(iter) ) {
    npy_int value = *(npy_int *)PyArray_ITER_DATA(iter);
    int index = N-1;
    int n = N;
    while ( n >>= 1 ) {
      int tryind = index^n;
      if ( tryind >= size || data[tryind] >= value ) {
        index = tryind;
      }
    }
    if ( index == size || data[index] != value ) {
      for ( i = size; i > index; i-- ) {
        data[i] = data[i-1];
      }
      data[index] = value;
      size++;
    }
    PyArray_ITER_NEXT( iter );
  }
  Py_DECREF( iter );
  Py_DECREF( entries );
  npy_intp stride = sizeof(npy_int);
  return new_dims_and_strides( base, 1, &size, &stride, NPY_WRITEABLE|NPY_CONTIGUOUS );
}

static PyObject *numeric_contract( PyObject *self, PyObject *args, PyObject *kwargs ) {
  // Contracts two array-like objects over specified axis int/axes tuple. Fully
  // equivalent with pointwise multiplication followed by summation:
  // (A*B).sum(n) == contract(A,B,n).

  PyObject *A = NULL, *B = NULL;
  int ncontract = -1;
  char *keywords[] = { "a", "b", "ncontract", NULL };
  if ( ! PyArg_ParseTupleAndKeywords( args, kwargs, "OOi", keywords, &A, &B, &ncontract ) ) {
    return NULL;
  }
  if ( ! PyArray_Check(A) || ! PyArray_Check(B) ) {
    PyErr_Format( PyExc_TypeError, "expected numpy arrays" );
    return NULL;
  }
  int nd = PyArray_NDIM(A);
  if ( nd != PyArray_NDIM(B) ) {
    PyErr_Format( PyExc_TypeError, "dimensions do not match" );
    return NULL;
  }
  if ( ncontract < 0 || ncontract > nd ) {
    PyErr_Format( PyExc_TypeError, "invalid ncontract" );
    return NULL;
  }
  DataStepper axes[NPY_MAXDIMS];
  int i, n;
  for ( i = 0; i < nd; i++ ) {
    n = PyArray_DIM(A,i);
    if ( n != PyArray_DIM(B,i) ) {
      PyErr_Format( PyExc_TypeError, "shapes do not match" );
      return NULL;
    }
    axes[i].countdown = axes[i].resetcounter = n-1;
    axes[i].stride0 = PyArray_STRIDE(A,i) / sizeof(double);
    axes[i].stride1 = PyArray_STRIDE(B,i) / sizeof(double);
  }
  PyObject *C = PyArray_EMPTY( nd-ncontract, PyArray_DIMS(A), NPY_DOUBLE, 0 );
  if ( C == NULL ) {
    return NULL;
  }
  double *ptrA = PyArray_DATA( A );
  double *ptrB = PyArray_DATA( B );
  double *ptrC = PyArray_DATA( C );
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

  { "addsorted",  (PyCFunction)numeric_addsorted,  METH_KEYWORDS, "add to sorted array" },
  { "contract",   (PyCFunction)numeric_contract,   METH_KEYWORDS, "contract"            },
};

static PyObject *numeric_module;
PyMODINIT_FUNC init_numeric( void ) {
  // Initializes module and numpy components.

  numeric_module = Py_InitModule( "_numeric", module_methods );
  import_array();
}

// vim:shiftwidth=2:foldmethod=syntax
