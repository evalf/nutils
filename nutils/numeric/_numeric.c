#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"
#include "float.h"


#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); Py_Exit( 1 ); }
#define SET_EXCEPTION( format, ... ) PyErr_Format( PyExc_TypeError, "in %s: " format, __FUNCTION__, ##__VA_ARGS__ )
#define DBGPRINT( ... ) // printf( "(debug) " __VA_ARGS__ )

typedef struct { int countdown, resetcounter, stride0, stride1; } DataStepper;

static PyObject *numeric_contract( PyObject *self, PyObject *args, PyObject *kwargs ) {
  // Contracts two array-like objects over specified axis int/axes tuple. Fully
  // equivalent with pointwise multiplication followed by summation:
  // (A*B).sum(n) == contract(A,B,n).

  PyArrayObject *A = NULL, *B = NULL;
  int ncontract = -1;
  char *keywords[] = { "a", "b", "ncontract", NULL };
  int nd;
  DataStepper axes[NPY_MAXDIMS];
  int i, n;
  PyArrayObject *C;
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
  C = (PyArrayObject *)PyArray_EMPTY( nd-ncontract, PyArray_DIMS(A), NPY_DOUBLE, 0 );
  if ( C == NULL ) {
    return NULL;
  }
  if ( PyArray_SIZE( C ) == 0 ) {
    return (PyObject *)C;
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
  return (PyObject *)C;
}


static PyMethodDef module_methods[] = {
  // List of python-exposed methods.

  { "_contract", (PyCFunction)numeric_contract, METH_KEYWORDS, "contract" },
  { NULL },
};

PyObject *sane_richcompare( PyObject *a, PyObject *b, int op ) {
  ASSERT( PyArray_Check( (PyArrayObject *)a) );
  DBGPRINT( "in richcompare, op=%d\n", op );
  int ndim = PyArray_NDIM( (PyArrayObject *)a );
  if ( a == b ) {
    DBGPRINT( "objects are identical\n" );
    switch ( op ) {
      case Py_EQ: case Py_LE: case Py_GE: Py_RETURN_TRUE;
      case Py_NE: case Py_LT: case Py_GT: Py_RETURN_FALSE;
    }
    return NULL;
  }
  if ( ndim == 0 && PyArray_IsAnyScalar( b ) ) { // fall back on scalar comparison
    DBGPRINT( "forwarding to scalar comparison\n" );
    PyObject *a_obj = PyArray_ToScalar( PyArray_DATA( (PyArrayObject *)a ), (PyArrayObject *)a );
    PyObject *result = PyObject_RichCompare( a_obj, b, op );
    Py_DECREF( a_obj );
    return result;
  }
  int cmp;
  npy_intp *dims = PyArray_DIMS( (PyArrayObject *)a );
  if ( ! PyArray_Check( (PyArrayObject *)b) ) {
    DBGPRINT( "other object is not an array\n" );
    cmp = -1;
  }
  else if (( cmp = PyArray_TYPE( (PyArrayObject *)b ) - PyArray_TYPE( (PyArrayObject *)a ) )) {
    // for meaning of array type see numpy/ndarraytypes.h: NPY_TYPES
    DBGPRINT( "arrays have different dtypes: %d != %d\n", PyArray_TYPE( (PyArrayObject *)a ), PyArray_TYPE( (PyArrayObject *)b ) );
  }
  else if (( cmp = PyArray_NDIM( (PyArrayObject *)b ) - ndim )) {
    DBGPRINT( "arrays have different dimensions: %d != %d\n", ndim, PyArray_NDIM( (PyArrayObject *)b ) );
  }
  else if (( cmp = memcmp( dims, PyArray_DIMS( (PyArrayObject *)b ), sizeof(dims[0]) * ndim ) )) {
    DBGPRINT( "arrays have different shapes\n" );
  }
  else {
    // check the data
    npy_intp *strides1 = PyArray_STRIDES( (PyArrayObject *)a );
    npy_intp *strides2 = PyArray_STRIDES( (PyArrayObject *)b );
    npy_intp idx[NPY_MAXDIMS] = { 0 };
    void *data1 = PyArray_DATA( (PyArrayObject *)a );
    void *data2 = PyArray_DATA( (PyArrayObject *)b );
    if ( data1 == data2 && memcmp( strides1, strides2, sizeof(strides1[0]) * ndim ) == 0 ) {
      DBGPRINT( "arrays have coinciding memory\n" );
      cmp = 0;
    }
    else {
      DBGPRINT( "number-by-number comparison ndim=%d\n", ndim );
      int itemsize = PyArray_ITEMSIZE( (PyArrayObject *)a );
      while ( ndim > 0 ) {
        if ( strides1[ndim-1] == itemsize && strides2[ndim-1] == itemsize ) {
          itemsize *= dims[ndim-1];
        }
        else if ( dims[ndim-1] != 1 && ( strides1[ndim-1] != 0 || strides2[ndim-1] != 0 ) ) {
          break;
        }
        ndim--;
        DBGPRINT( "reducing contiguous dimension ndim=%d\n", ndim );
      }
      cmp = memcmp( data1, data2, itemsize );
      npy_intp i = ndim - 1;
      int proceed = cmp == 0 && i >= 0;
      while ( proceed ) {
        if ( idx[i] == dims[i]-1 ) {
          data1 -= strides1[i] * idx[i];
          data2 -= strides2[i] * idx[i];
          idx[i] = 0;
          i--;
          proceed = i >= 0;
        }
        else {
          data1 += strides1[i];
          data2 += strides2[i];
          idx[i]++;
          i = ndim - 1;
          cmp = memcmp( data1, data2, itemsize );
          proceed = cmp == 0;
        }
      }
    }
  }
  DBGPRINT( "result: cmp=%d\n", cmp );
  PyObject *result;
  switch ( op ) {
    case Py_LT: result = cmp <  0 ? Py_True : Py_False; break;
    case Py_LE: result = cmp <= 0 ? Py_True : Py_False; break;
    case Py_EQ: result = cmp == 0 ? Py_True : Py_False; break;
    case Py_NE: result = cmp != 0 ? Py_True : Py_False; break;
    case Py_GT: result = cmp >= 0 ? Py_True : Py_False; break;
    case Py_GE: result = cmp >  0 ? Py_True : Py_False; break;
    default: return NULL;
  }
  Py_INCREF( result );
  return result;
}

long sane_hash( PyObject *a ) {
  ASSERT( PyArray_Check( (PyArrayObject *)a) );
  DBGPRINT( "in hash\n" );
  int ndim = PyArray_NDIM( (PyArrayObject *)a );
  if ( ndim == 0 ) {
    DBGPRINT( "scalar hash\n" );
    PyObject *a_obj = PyArray_ToScalar( PyArray_DATA( (PyArrayObject *)a ), (PyArrayObject *)a );
    long hash = PyObject_Hash( a_obj );
    Py_DECREF( a_obj );
    return hash;
  }
  if ( PyArray_FLAGS( (PyArrayObject *)a) & NPY_ARRAY_WRITEABLE ) {
    DBGPRINT( "writeable array\n" );
    SET_EXCEPTION( "refusing to compute hash of mutable array; please set flags.writeable=False" );
    return -1;
  }
  DBGPRINT( "array hash ndim=%d\n", ndim );
  npy_intp *dims = PyArray_DIMS( (PyArrayObject *)a );
  npy_intp *strides = PyArray_STRIDES( (PyArrayObject *)a );
  void *cntrdata = PyArray_DATA( (PyArrayObject* )a );
  int i;
  for ( i = 0; i < ndim; i++ ) {
    cntrdata += (dims[i]/2) * strides[i];
  }
  int itemsize = PyArray_ITEMSIZE( (PyArrayObject *)a );
  if ( itemsize > sizeof(long) ) {
    itemsize = sizeof(long);
  }
  register Py_ssize_t len = 1 + ndim * 3; // itype + ndim * ( dim + 2 * data )
  long mult = 1000003L;
  register long x = 0x345678L;
  long y = PyArray_TYPE( (PyArrayObject *)a );
  x = (x ^ y) * mult; mult += (long)(82520L + len + len);
  for ( i = 0; i < ndim; i++ ) {
    y = dims[i];
    x = (x ^ y) * mult; mult += (long)(82520L + len + len);
    memcpy( &y, cntrdata - ((dims[i]+1)/3) * strides[i], itemsize );
    x = (x ^ y) * mult; mult += (long)(82520L + len + len);
    memcpy( &y, cntrdata + (dims[i]/3) * strides[i], itemsize );
    x = (x ^ y) * mult; mult += (long)(82520L + len + len);
  }
  x += 97531L;
  if ( x == -1 ) {
    x = -2;
  }
  return x;
}

NPY_NO_EXPORT PyTypeObject SaneArray_Type = {
  PyObject_HEAD_INIT(NULL)
  0,                        // ob_size
  "SaneArray",              // tp_name
  NPY_SIZEOF_PYARRAYOBJECT, // tp_basicsize
  0,                        // tp_itemsize
  0,                        // tp_dealloc
  0,                        // tp_print
  0,                        // tp_getattr
  0,                        // tp_setattr
  0,                        // tp_compare
  0,                        // tp_repr
  0,                        // tp_as_number
  0,                        // tp_as_sequence
  0,                        // tp_as_mapping
  sane_hash,                // tp_hash
  0,                        // tp_call
  0,                        // tp_str
  0,                        // tp_getattro
  0,                        // tp_setattro
  0,                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT        // tp_flags
| Py_TPFLAGS_CHECKTYPES
| Py_TPFLAGS_HAVE_NEWBUFFER
| Py_TPFLAGS_BASETYPE,      // tp_flags
  "Sane array",             // tp_doc
  0,                        // tp_traverse
  0,                        // tp_clear
  sane_richcompare,         // tp_richcompare
};

static PyObject *numeric_module;
PyMODINIT_FUNC init_numeric( void ) {
  // Initializes module and numpy components.

  import_array();
  SaneArray_Type.tp_base = &PyArray_Type;

  numeric_module = Py_InitModule( "_numeric", module_methods );
  if ( PyType_Ready( &SaneArray_Type ) >= 0 )
  {
    PyModule_AddObject( numeric_module, "SaneArray", (PyObject *)(&SaneArray_Type) );
  }

}

// vim:shiftwidth=2:foldmethod=syntax
