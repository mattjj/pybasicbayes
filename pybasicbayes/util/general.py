from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import zip
from builtins import range
import sys
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import scipy.linalg
import scipy.linalg.lapack as lapack
import copy, collections, os, shutil, hashlib
from contextlib import closing
from itertools import chain, count
from functools import reduce
from urllib.request import urlopen  # py2.7 covered by standard_library.install_aliases()


def blockarray(*args,**kwargs):
    return np.array(np.bmat(*args,**kwargs),copy=False)

def interleave(*iterables):
    return list(chain.from_iterable(zip(*iterables)))

def joindicts(dicts):
    # stuff on right clobbers stuff on left
    return reduce(lambda x,y: dict(x,**y), dicts, {})

def one_vs_all(stuff):
    stuffset = set(stuff)
    for thing in stuff:
        yield thing, stuffset - set([thing])

def rle(stateseq):
    pos, = np.where(np.diff(stateseq) != 0)
    pos = np.concatenate(([0],pos+1,[len(stateseq)]))
    return stateseq[pos[:-1]], np.diff(pos)

def irle(vals,lens):
    out = np.empty(np.sum(lens))
    for v,l,start in zip(vals,lens,np.concatenate(((0,),np.cumsum(lens)[:-1]))):
        out[start:start+l] = v
    return out

def ibincount(counts):
    'returns an array a such that counts = np.bincount(a)'
    return np.repeat(np.arange(counts.shape[0]),counts)

def cumsum(v,strict=False):
    if not strict:
        return np.cumsum(v,axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1],axis=0)
        return out

def rcumsum(v,strict=False):
    if not strict:
        return np.cumsum(v[::-1],axis=0)[::-1]
    else:
        out = np.zeros_like(v)
        out[:-1] = np.cumsum(v[-1:0:-1],axis=0)[::-1]
        return out

def delta_like(v,i):
    out = np.zeros_like(v)
    out[i] = 1
    return out

def deepcopy(obj):
    return copy.deepcopy(obj)

def nice_indices(arr):
    '''
    takes an array like [1,1,5,5,5,999,1,1]
    and maps to something like [0,0,1,1,1,2,0,0]
    modifies original in place as well as returns a ref
    '''
    # surprisingly, this is slower for very small (and very large) inputs:
    # u,f,i = np.unique(arr,return_index=True,return_inverse=True)
    # arr[:] = np.arange(u.shape[0])[np.argsort(f)][i]
    ids = collections.defaultdict(count().__next__)
    for idx,x in enumerate(arr):
        arr[idx] = ids[x]
    return arr

def ndargmax(arr):
    return np.unravel_index(np.argmax(np.ravel(arr)),arr.shape)

def match_by_overlap(a,b):
    assert a.ndim == b.ndim == 1 and a.shape[0] == b.shape[0]
    ais, bjs = list(set(a)), list(set(b))
    scores = np.zeros((len(ais),len(bjs)))
    for i,ai in enumerate(ais):
        for j,bj in enumerate(bjs):
            scores[i,j] = np.dot(np.array(a==ai,dtype=np.float),b==bj)

    flip = len(bjs) > len(ais)

    if flip:
        ais, bjs = bjs, ais
        scores = scores.T

    matching = []
    while scores.size > 0:
        i,j = ndargmax(scores)
        matching.append((ais[i],bjs[j]))
        scores = np.delete(np.delete(scores,i,0),j,1)
        ais = np.delete(ais,i)
        bjs = np.delete(bjs,j)

    return matching if not flip else [(x,y) for y,x in matching]

def hamming_error(a,b):
    return (a!=b).sum()

def scoreatpercentile(data,per,axis=0):
    'like the function in scipy.stats but with an axis argument and works on arrays'
    a = np.sort(data,axis=axis)
    idx = per/100. * (data.shape[axis]-1)

    if (idx % 1 == 0):
        return a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]]
    else:
        lowerweight = 1-(idx % 1)
        upperweight = (idx % 1)
        idx = int(np.floor(idx))
        return lowerweight * a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]] \
                + upperweight * a[[slice(None) if ii != axis else idx+1 for ii in range(a.ndim)]]

def stateseq_hamming_error(sampledstates,truestates):
    sampledstates = np.array(sampledstates,ndmin=2).copy()

    errors = np.zeros(sampledstates.shape[0])
    for idx,s in enumerate(sampledstates):
        # match labels by maximum overlap
        matching = match_by_overlap(s,truestates)
        s2 = s.copy()
        for i,j in matching:
            s2[s==i] = j
        errors[idx] = hamming_error(s2,truestates)

    return errors if errors.shape[0] > 1 else errors[0]

def _sieve(stream):
    # just for fun; doesn't work over a few hundred
    val = next(stream)
    yield val
    for x in [x for x in _sieve(stream) if x % val != 0]:
        yield x

def primes():
    return _sieve(count(2))

def top_eigenvector(A,niter=1000,force_iteration=False):
    '''
    assuming the LEFT invariant subspace of A corresponding to the LEFT
    eigenvalue of largest modulus has geometric multiplicity of 1 (trivial
    Jordan block), returns the vector at the intersection of that eigenspace and
    the simplex

    A should probably be a ROW-stochastic matrix

    probably uses power iteration
    '''
    n = A.shape[0]
    np.seterr(invalid='raise',divide='raise')
    if n <= 25 and not force_iteration:
        x = np.repeat(1./n,n)
        x = np.linalg.matrix_power(A.T,niter).dot(x)
        x /= x.sum()
        return x
    else:
        x1 = np.repeat(1./n,n)
        x2 = x1.copy()
        for itr in range(niter):
            np.dot(A.T,x1,out=x2)
            x2 /= x2.sum()
            x1,x2 = x2,x1
            if np.linalg.norm(x1-x2) < 1e-8:
                break
        return x1

def engine_global_namespace(f):
    # see IPython.parallel.util.interactive; it's copied here so as to avoid
    # extra imports/dependences elsewhere, and to provide a slightly clearer
    # name
    f.__module__ = '__main__'
    return f

def block_view(a,block_shape):
    shape = (a.shape[0]/block_shape[0],a.shape[1]/block_shape[1]) + block_shape
    strides = (a.strides[0]*block_shape[0],a.strides[1]*block_shape[1]) + a.strides
    return ast(a,shape=shape,strides=strides)

def AR_striding(data,nlags):
    data = np.asarray(data)
    if not data.flags.c_contiguous:
        data = data.copy(order='C')
    if data.ndim == 1:
        data = np.reshape(data,(-1,1))
    sz = data.dtype.itemsize
    return ast(
            data,
            shape=(data.shape[0]-nlags,data.shape[1]*(nlags+1)),
            strides=(data.shape[1]*sz,sz))

def count_transitions(stateseq,minlength=None):
    if minlength is None:
        minlength = stateseq.max() + 1
    out = np.zeros((minlength,minlength),dtype=np.int32)
    for a,b in zip(stateseq[:-1],stateseq[1:]):
        out[a,b] += 1
    return out

### SGD

def sgd_steps(tau,kappa):
    assert 0.5 < kappa <= 1 and tau >= 0
    for t in count(1):
        yield (t+tau)**(-kappa)

def hold_out(datalist,frac):
    N = len(datalist)
    perm = np.random.permutation(N)
    split = int(np.ceil(frac * N))
    return [datalist[i] for i in perm[split:]], [datalist[i] for i in perm[:split]]

def sgd_passes(tau,kappa,datalist,minibatchsize=1,npasses=1):
    N = len(datalist)

    for superitr in range(npasses):
        if minibatchsize == 1:
            perm = np.random.permutation(N)
            for idx, rho_t in zip(perm,sgd_steps(tau,kappa)):
                yield datalist[idx], rho_t
        else:
            minibatch_indices = np.array_split(np.random.permutation(N),N/minibatchsize)
            for indices, rho_t in zip(minibatch_indices,sgd_steps(tau,kappa)):
                yield [datalist[idx] for idx in indices], rho_t

def sgd_sampling(tau,kappa,datalist,minibatchsize=1):
    N = len(datalist)
    if minibatchsize == 1:
        for rho_t in sgd_steps(tau,kappa):
            minibatch_index = np.random.choice(N)
            yield datalist[minibatch_index], rho_t
    else:
        for rho_t in sgd_steps(tau,kappa):
            minibatch_indices = np.random.choice(N,size=minibatchsize,replace=False)
            yield [datalist[idx] for idx in minibatch_indices], rho_t

# TODO should probably eliminate this function
def minibatchsize(lst):
    return float(sum(d.shape[0] for d in lst))

### misc

def random_subset(lst,sz):
    perm = np.random.permutation(len(lst))
    return [lst[perm[idx]] for idx in range(sz)]

def get_file(remote_url,local_path):
    if not os.path.isfile(local_path):
        with closing(urlopen(remote_url)) as remotefile:
            with open(local_path,'wb') as localfile:
                shutil.copyfileobj(remotefile,localfile)

def list_split(lst,num):
    assert num > 0
    return [lst[start::num] for start in range(num)]

def ndarrayhash(v):
    assert isinstance(v,np.ndarray)
    return hashlib.sha1(v).hexdigest()

### numerical linear algebra

def inv_psd(A, return_chol=False):
    L = np.linalg.cholesky(A)
    Ainv = lapack.dpotri(L, lower=True)[0]
    copy_lower_to_upper(Ainv)
    # if not np.allclose(Ainv, np.linalg.inv(A), rtol=1e-5, atol=1e-5):
    #     import ipdb; ipdb.set_trace()
    if return_chol:
        return Ainv, L
    else:
        return Ainv

def solve_psd(A,b,chol=None,lower=True,overwrite_b=False,overwrite_A=False):
    if chol is None:
        return lapack.dposv(A,b,overwrite_b=overwrite_b,overwrite_a=overwrite_A)[1]
    else:
        return lapack.dpotrs(chol,b,lower,overwrite_b)[0]

def copy_lower_to_upper(A):
    A += np.tril(A,k=-1).T


# NOTE: existing numpy object array construction acts a bit weird, e.g.
# np.array([randn(3,4),randn(3,5)]) vs np.array([randn(3,4),randn(5,3)])
# this wrapper class is just meant to ensure that when ndarrays of objects are
# constructed the construction doesn't "recurse" as in the first example
class ObjArray(np.ndarray):
    def __new__(cls,lst):
        if isinstance(lst,(np.ndarray,float,int)):
            return lst
        else:
            return np.ndarray.__new__(cls,len(lst),dtype=np.object)

    def __init__(self,lst):
        if not isinstance(lst,(np.ndarray,float,int)):
            for i, elt in enumerate(lst):
                self[i] = self.__class__(elt)

# Here's an alternative to ObjArray: just construct an obj array from a list
def objarray(lst):
    a = np.empty(len(lst), dtype=object)
    for i,o in enumerate(lst):
        a[i] = o
    return a

def all_none(*args):
    return all(_ is None for _ in args)

def any_none(*args):
    return any(_ is None for _ in args)

