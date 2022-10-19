"""
The sparse module defines a dtype for numpy that represents sparse data in
n-dimensional coo-format. That is, every array element contains an index into a
larger sparse object, and a value, which can be of any numpy supported data
type including integers, floating point values and complex data. Additionally,
the dtype carries the shape of the sparse object as metadata, which makes the
numpy array into an entirely self contained sparse object.

In addition to the dtype, the sparse module provides a range of methods for
manipulation of sparse data, such as deduplication of indices, pruning of
zeros, sparse addition, and conversion to other sparse or dense data formats.
"""

import numpy

chunksize = 0x10000000  # 256MB


def dtype(shape, vtype=numpy.float64):
    '''Numpy data dtype for sparse data.

    Returns a structured dtype with fields 'index' and 'value', where index is
    again structured with fields 'i0', 'i1', etc, and value is of type ``vtype``.
    The indices are of the smallest unsigned integer type that can encode all
    indices within ``shape``, and carry the shape as metadata.

    Args
    ----
    shape : :class:`tuple` of integers.
        Shape of the sparse object.
    vtype : :class:`numpy.dtype` or :class:`str`
        Data dype of the sparse object (i.e. the nonzero values).

    Returns
    -------
    dtype : :class:`numpy.dtype`
        The sparse dtype.
    '''

    return _dtype([((int(n), 'i'+str(i)), _uint(n)) for i, n in enumerate(shape)], vtype)


def issparse(data):
    return isinstance(data, numpy.ndarray) and issparsedtype(data.dtype)


def issparsedtype(dtype):
    return dtype.names == ('index', 'value') and all(
        len(value) == 3 and isinstance(value[2], int) and 0 <= value[2] <= 256**value[0].itemsize
        for value in dtype['index'].fields.values())


def ndim(data):
    '''Dimension of the sparse object.'''

    return len(data.dtype['index'].names)


def shape(data):
    '''Shape of the sparse object.'''

    itype = data.dtype['index']
    return tuple(itype.fields[name][2] for name in itype.names)


def indices(data):
    '''Tuple of indices of the nonzero values of the sparse object.'''

    index = data['index']
    return tuple(index[i] for i in index.dtype.names)


def values(data):
    '''Nonzero values of the sparse object.'''

    return data['value']


def extract(data):
    '''Tuple of indices, values, and shape of the sparse object.'''

    return indices(data), values(data), shape(data)


def empty(shape, vtype=numpy.float64):
    '''Completely sparse array of given shape and data type.'''

    return numpy.empty(0, dtype=dtype(shape, vtype))


def result_type(dtype0, *dtypes):
    '''Sparse analogue of :func:`numpy.result_type`.'''

    if not dtypes:
        return dtype0
    if any(dtype['index'] != dtype0['index'] for dtype in dtypes):
        raise Exception('non-matching shapes')
    return _dtype(dtype0['index'], numpy.result_type(dtype0['value'], *[dtype['value'] for dtype in dtypes]))


def dedup(data, inplace=False):
    '''Deduplicate indices.

    Dedup sorts data in lexicographical order and sums all values with matching
    indices such that the returned array has at most one value per sparse index.
    The sorting happens in place, which means that ``dedup`` changes the order of
    the input argument. Additionally, if ``inplace`` is true, the deduplication
    step reuses the input array's memory. This may affect the size of the array,
    which should no longer be used after deduplication in place. In case the
    input has no duplicates the input array is returned.

    >>> from nutils.sparse import dtype, dedup
    >>> from numpy import array
    >>> A = array([((0,1),.1), ((1,0),.2), ((0,1),.3)], dtype=dtype([2,2]))
    >>> dedup(A)
    array([((0, 1),  0.4), ((1, 0),  0.2)],
          dtype=[('index', [((2, 'i0'), 'u1'), ((2, 'i1'), 'u1')]), ('value', '<f8')])
    '''

    if not len(data):
        return data
    if not ndim(data):
        return data['value'].sum()[numpy.newaxis].view(data.dtype)
    data.view(numpy.void).sort(kind='stable')  # stable = timsort
    keep = data['index'][1:] != data['index'][:-1]
    if keep.all():
        return data
    elif inplace:
        buf = numpy.empty(chunksize // data.dtype.itemsize or 1, dtype=data.dtype)
        n, = numpy.hstack([True, keep, True]).nonzero()
        for i in range(0, len(n)-1, len(buf)):
            s = numpy.diff(n[i:i+len(buf)+1])
            overlap = i+len(s) > n[i]
            chunk = buf[:len(s)] if overlap else data[i:i+len(s)]
            numpy.take(data['index'], n[i:i+len(s)], out=chunk['index'])
            chunk['value'].fill(0)
            numpy.add.at(chunk['value'], numpy.arange(len(s)).repeat(s), data['value'][n[i]:n[i+len(s)]])
            if overlap:
                data[i:i+len(s)] = chunk
        return _resize(data, len(n)-1)
    else:
        offsets = keep.cumsum()
        dedup = numpy.empty(offsets[-1]+1, dtype=data.dtype)
        dedup[0] = data[0]
        numpy.compress(keep, data['index'][1:], out=dedup['index'][1:])
        dedup['value'][1:].fill(0)
        numpy.add.at(dedup['value'], offsets, data['value'][1:])
        return dedup


def prune(data, inplace=False, mask=None):
    '''Prune zero values.

    Prune returns a sparse object with all zero values removed, or all entries
    for which the boolean vector ``mask`` is true if it is specified. If
    ``inplace`` is true the returned object reuses the input array's memory. This
    may affect the size of the array, which should no longer be used after
    pruning in place. In case the input has no zeros the input array is returned.

    >>> from nutils.sparse import dtype, prune
    >>> from numpy import array
    >>> A = array([((0,1),.1), ((1,0),0), ((0,1),.3)], dtype=dtype([2,2]))
    >>> prune(A)
    array([((0, 1),  0.1), ((0, 1),  0.3)],
          dtype=[('index', [((2, 'i0'), 'u1'), ((2, 'i1'), 'u1')]), ('value', '<f8')])
    '''

    if mask is None:
        mask = data['value']
    if mask.all():
        return data
    elif inplace:
        buf = numpy.empty(chunksize // data.dtype.itemsize or 1, dtype=data.dtype)
        nz, = mask.nonzero()
        for i in range(0, len(nz), len(buf)):
            s = nz[i:i+len(buf)]
            overlap = i+len(s) > s[0]
            chunk = buf[:len(s)] if overlap else data[i:i+len(s)]
            numpy.take(data, s, out=chunk)
            if overlap:
                data[i:i+len(s)] = chunk
        return _resize(data, len(nz))
    else:
        return numpy.compress(mask, data)


def add(datas):
    '''Add sparse objects.

    Returns the sum of a list of sparse objects by concatenating the sparse
    entries. The returned array is of the data type mandated by Numpy's promotion
    rules. In case ``datas`` contains only one item of nonzero length and this
    item has the correct data type, then this array is returned as-is.

    >>> from nutils.sparse import dtype, add
    >>> from numpy import array
    >>> A = array([((0,1),.1), ((1,0),.2)], dtype=dtype([2,2]))
    >>> B = array([((0,1),.3)], dtype=dtype([2,2]))
    >>> add([A, B])
    array([((0, 1),  0.1), ((1, 0),  0.2), ((0, 1),  0.3)],
          dtype=[('index', [((2, 'i0'), 'u1'), ((2, 'i1'), 'u1')]), ('value', '<f8')])
    '''

    dtype = result_type(*[data.dtype for data in datas])
    datas = [data for data in datas if len(data)]
    if len(datas) == 1 and datas[0].dtype == dtype:
        return datas[0]
    retval = numpy.empty(sum(map(len, datas)), dtype=dtype)
    if datas:
        numpy.concatenate(datas, out=retval)
    return retval


def block(datas):
    '''Stack sparse blocks.'''

    assert isinstance(datas, list)
    structure = [len(datas)]
    while all(isinstance(data, list) for data in datas):
        length = set(map(len, datas))
        if len(length) != 1:
            raise Exception('inconsistent block structure')
        datas = sum(datas, [])
        structure.extend(length)
    blocksizes = [numpy.repeat(-1, sh) for sh in structure]
    blocks = []
    for i, data in enumerate(datas):
        if isinstance(data, int) and data == 0:
            continue
        if not issparse(data) or ndim(data) != len(structure):
            raise Exception('all blocks should be either {}d-sparse or 0'.format(len(structure)))
        blockidx = numpy.unravel_index(i, structure)
        for blocksize, iblock, size in zip(blocksizes, blockidx, shape(data)):
            if blocksize[iblock] == -1:
                blocksize[iblock] = size
            elif blocksize[iblock] != size:
                raise Exception('block sizes do not match')
        blocks.append((blockidx, data))
    if any(-1 in blocksize for blocksize in blocksizes):
        raise Exception('not all block sizes can be determined')
    retval = numpy.empty(sum(len(data) for blockidx, data in blocks),
                         dtype=dtype([blocksize.sum() for blocksize in blocksizes], numpy.result_type(*[data.dtype['value'] for blockidx, data in blocks])))
    i = 0
    for blockidx, data in blocks:
        j = i + len(data)
        out = retval[i:j]
        out['value'] = data['value']
        for idim, iblock in enumerate(blockidx):
            outindex = out['index']['i'+str(idim)]
            numpy.add(data['index']['i'+str(idim)], blocksizes[idim][:iblock].sum(), out=outindex, dtype=outindex.dtype.type)
            # NOTE: Specifying the dtype is necessary because Numpy apparently does
            # not follow the out argument automatically, defaulting instead to a
            # dtype that may be to small to contain the result of the addition.
        i = j
    assert i == len(retval)
    return retval


def take(data, select):
    '''Take a multidimensional, ordered subset.'''

    indices, values, shape = extract(data)
    newindices = []
    newshape = []
    keep = True
    axis = 0
    for s in select:
        if not isinstance(s, numpy.ndarray) or s.dtype != bool:
            raise Exception('invalid selection vector for dimension {}'.format(axis))
        nextaxis = axis + s.ndim
        if s.shape != shape[axis:nextaxis]:
            raise Exception('selection vector has incorrect shape for dimension {}'.format(axis))
        sh = s.sum()
        if sh != s.size:
            keep &= s[indices[axis:nextaxis]]
            renumber = numpy.empty(s.shape, dtype=_uint(sh))
            renumber[s] = numpy.arange(sh)
            index = renumber[indices[axis:nextaxis]]
            del renumber
        elif s.ndim == 1:
            index = indices[axis]
        else:
            index = numpy.zeros(len(data), dtype=_uint(sh))
            for i in range(axis, nextaxis):
                index *= shape[i]
                index += indices[i]
        newindices.append(index)
        newshape.append(sh)
        axis = nextaxis
    assert axis == len(shape)
    if keep is not True:  # selection
        retval = numpy.empty(keep.sum(), dtype=dtype(newshape, data['value'].dtype))
        numpy.compress(keep, data['value'], out=retval['value'])
        for i, newindex in enumerate(newindices):
            numpy.compress(keep, newindex, out=retval['index']['i'+str(i)])
    elif newshape != list(shape):  # reshape
        retval = numpy.empty(len(data), dtype=dtype(newshape, data['value'].dtype))
        retval['value'] = data['value']
        for i, newindex in enumerate(newindices):
            retval['index']['i'+str(i)] = newindex
    else:  # identity
        retval = data
    return retval


def toarray(data):
    '''Convert sparse object to a dense array.

    >>> from nutils.sparse import dtype, toarray
    >>> from numpy import array
    >>> A = array([((0,1),.1), ((1,0),.2), ((0,1),.3)], dtype=dtype([2,2]))
    >>> toarray(A)
    array([[ 0. ,  0.4],
           [ 0.2,  0. ]])
    '''

    indices, values, shape = extract(data)
    if not shape:
        return values.sum()
    if len(shape) == 1 and indices[0].shape[0] == shape[0] and (indices[0][:-1] < indices[0][1:]).all():
        return values.copy()
    retval = numpy.zeros(shape, values.dtype)
    numpy.add.at(retval, indices, values)
    return retval


def fromarray(data):
    '''Convert dense array to sparse object.

    >>> from nutils.sparse import dtype, fromarray
    >>> from numpy import array
    >>> A = array([[0, .4], [.2, 0]])
    >>> fromarray(A)
    array([((0, 0),  0. ), ((0, 1),  0.4), ((1, 0),  0.2), ((1, 1),  0. )],
          dtype=[('index', [((2, 'i0'), 'u1'), ((2, 'i1'), 'u1')]), ('value', '<f8')])
    '''

    retval = numpy.empty(data.size, dtype=dtype(data.shape, data.dtype))
    retval.reshape(data.shape)['value'] = data
    index = retval.reshape(data.shape)['index']
    for i, sh in enumerate(data.shape):
        index['i'+str(i)] = numpy.arange(sh).reshape([-1]+[1]*(data.ndim-i-1))
    return retval

# internal methods


def _dtype(itype, vtype):
    return numpy.dtype([('index', itype), ('value', vtype)])


def _uint(n):
    return numpy.dtype('>u'+str(1 if n <= 256 else 2 if n <= 256**2 else 4 if n <= 256**4 else 8))


def _resize(data, n):
    if data.base is not None:
        return data[:n]
    data.resize(n, refcheck=False)
    return data

# vim:sw=4:sts=4:et
