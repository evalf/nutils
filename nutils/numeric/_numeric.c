#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"
#include "float.h"


#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); Py_Exit( 1 ); }
#define SET_EXCEPTION( format, ... ) PyErr_Format( PyExc_TypeError, "in %s: " format, __FUNCTION__, ##__VA_ARGS__ )
#define DBGPRINT( ... ) // printf( "(debug) " __VA_ARGS__ )

// NUMERIC ARRAY

NPY_NO_EXPORT PyTypeObject NumericArray_Type;

PyObject *numeric_richcompare( PyArrayObject *self, PyObject *other, int op ) {
  DBGPRINT( "in richcompare, op=%d\n", op );
  if ( self == (PyArrayObject *)other ) {
    DBGPRINT( "objects are identical\n" );
    switch ( op ) {
      case Py_EQ: case Py_LE: case Py_GE: Py_RETURN_TRUE;
      case Py_NE: case Py_LT: case Py_GT: Py_RETURN_FALSE;
    }
    return NULL;
  }
  int ndim = PyArray_NDIM(self);
  if ( ndim == 0 ) { // fall back on scalar comparison
    DBGPRINT( "forwarding to scalar comparison\n" );
    PyObject *a_obj = PyArray_DESCR(self)->f->getitem( PyArray_DATA(self), self );
    PyObject *result = PyObject_RichCompare( a_obj, other, op );
    Py_DECREF( a_obj );
    return result;
  }
  int cmp;
  npy_intp *dims = PyArray_DIMS(self);
  if ( ! PyArray_Check( (PyArrayObject *)other) ) {
    DBGPRINT( "other object is not an array\n" );
    cmp = -1;
  }
  else if (( cmp = PyArray_TYPE( (PyArrayObject *)other ) - PyArray_TYPE(self) )) {
    // for meaning of array type see numpy/ndarraytypes.h: NPY_TYPES
    DBGPRINT( "arrays have different dtypes: %d != %d\n", PyArray_TYPE(self), PyArray_TYPE( (PyArrayObject *)other ) );
  }
  else if (( cmp = PyArray_NDIM( (PyArrayObject *)other ) - ndim )) {
    DBGPRINT( "arrays have different dimensions: %d != %d\n", ndim, PyArray_NDIM( (PyArrayObject *)other ) );
  }
  else if (( cmp = memcmp( dims, PyArray_DIMS( (PyArrayObject *)other ), sizeof(dims[0]) * ndim ) )) {
    DBGPRINT( "arrays have different shapes\n" );
  }
  else {
    // check the data
    npy_intp *strides1 = PyArray_STRIDES(self);
    npy_intp *strides2 = PyArray_STRIDES( (PyArrayObject *)other );
    npy_intp idx[NPY_MAXDIMS] = { 0 };
    void *data1 = PyArray_DATA(self);
    void *data2 = PyArray_DATA( (PyArrayObject *)other );
    if ( data1 == data2 && memcmp( strides1, strides2, sizeof(strides1[0]) * ndim ) == 0 ) {
      DBGPRINT( "arrays have coinciding memory\n" );
      cmp = 0;
    }
    else {
      DBGPRINT( "number-by-number comparison ndim=%d\n", ndim );
      int itemsize = PyArray_ITEMSIZE(self);
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

long numeric_hash( PyArrayObject *self ) {
  DBGPRINT( "in hash\n" );
  int ndim = PyArray_NDIM(self);
  if ( ndim == 0 ) {
    DBGPRINT( "scalar hash\n" );
    PyObject *a_obj = PyArray_DESCR(self)->f->getitem( PyArray_DATA(self), self );
    long hash = PyObject_Hash( a_obj );
    Py_DECREF( a_obj );
    return hash;
  }
  DBGPRINT( "array hash ndim=%d\n", ndim );
  npy_intp *dims = PyArray_DIMS(self);
  npy_intp *strides = PyArray_STRIDES(self);
  void *cntrdata = PyArray_DATA(self);
  int i;
  for ( i = 0; i < ndim; i++ ) {
    cntrdata += (dims[i]/2) * strides[i];
  }
  int itemsize = PyArray_ITEMSIZE(self);
  if ( itemsize > sizeof(long) ) {
    itemsize = sizeof(long);
  }
  register Py_ssize_t len = 1 + ndim * 3; // itype + ndim * ( dim + 2 * data )
  long mult = 1000003L;
  register long x = 0x345678L;
  long y = PyArray_TYPE(self);
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

NPY_NO_EXPORT PyTypeObject NumericArray_Type = {
  PyObject_HEAD_INIT(NULL)
  0, // ob_size
  "NumericArray", // tp_name
  NPY_SIZEOF_PYARRAYOBJECT, // tp_basicsize
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  (hashfunc)numeric_hash, // tp_hash
  0, 0, 0, 0, 0,
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_HAVE_NEWBUFFER | Py_TPFLAGS_BASETYPE, // tp_flags
  "Numeric array", // tp_doc
  0, 0,
  (richcmpfunc)numeric_richcompare, // tp_richcompare
};

// NUMERIC MODULE

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
  C = (PyArrayObject *)PyArray_New( &NumericArray_Type, nd-ncontract, PyArray_DIMS(A), NPY_DOUBLE, NULL, NULL, 0, 0, NULL );
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

static PyObject *numeric_module;
PyMODINIT_FUNC init_numeric( void ) {
  // Initializes module and numpy components.

  import_array();
  NumericArray_Type.tp_base = &PyArray_Type;
  NumericArray_Type.tp_base = &PyArray_Type;

  numeric_module = Py_InitModule( "_numeric", module_methods );
  if ( PyType_Ready( &NumericArray_Type ) >= 0 )
  {
    PyModule_AddObject( numeric_module, "NumericArray", (PyObject *)(&NumericArray_Type) );
  }
  if ( PyType_Ready( &NumericArray_Type ) >= 0 )
  {
    PyModule_AddObject( numeric_module, "NumericArray", (PyObject *)(&NumericArray_Type) );
  }

}

// vim:shiftwidth=2:foldmethod=syntax
