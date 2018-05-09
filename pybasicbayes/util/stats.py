from __future__ import division
from __future__ import absolute_import
from builtins import range
import numpy as np
from numpy.random import random
na = np.newaxis
import scipy.stats as stats
import scipy.special as special
import scipy.linalg
from scipy.misc import logsumexp
from numpy.core.umath_tests import inner1d

from .general import any_none, blockarray

### data abstraction

# the data type is ndarrays OR lists of ndarrays
# type Data = ndarray | [ndarray]

def atleast_2d(data):
    # NOTE: can't use np.atleast_2d because if it's 1D we want axis 1 to be the
    # singleton and axis 0 to be the sequence index
    if data.ndim == 1:
        return data.reshape((-1,1))
    return data

def mask_data(data):
    return np.ma.masked_array(
        np.nan_to_num(data),np.isnan(data),fill_value=0.,hard_mask=True)

def gi(data):
    out = (np.isnan(atleast_2d(data)).sum(1) == 0).ravel()
    return out if len(out) != 0 else None

def getdatasize(data):
    if isinstance(data,np.ma.masked_array):
        return data.shape[0] - data.mask.reshape((data.shape[0],-1))[:,0].sum()
    elif isinstance(data,np.ndarray):
        if len(data) == 0:
            return 0
        return data[gi(data)].shape[0]
    elif isinstance(data,list):
        return sum(getdatasize(d) for d in data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def getdatadimension(data):
    if isinstance(data,np.ndarray):
        assert data.ndim > 1
        return data.shape[1]
    elif isinstance(data,list):
        assert len(data) > 0
        return getdatadimension(data[0])
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return 1

def combinedata(datas):
    ret = []
    for data in datas:
        if isinstance(data,np.ma.masked_array):
            ret.append(np.ma.compress_rows(data))
        if isinstance(data,np.ndarray):
            ret.append(data)
        elif isinstance(data,list):
            ret.extend(combinedata(data))
        else:
            # handle unboxed case for convenience
            assert isinstance(data,int) or isinstance(data,float)
            ret.append(np.atleast_1d(data))
    return ret

def flattendata(data):
    # data is either an array (possibly a maskedarray) or a list of arrays
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,list) or isinstance(data,tuple):
        if any(isinstance(d,np.ma.MaskedArray) for d in data):
            return np.concatenate([np.ma.compress_rows(d) for d in data])
        else:
            return np.concatenate(data)
    else:
        # handle unboxed case for convenience
        assert isinstance(data,int) or isinstance(data,float)
        return np.atleast_1d(data)

### misc
def update_param(oldv, newv, stepsize):
    return oldv * (1 - stepsize) + newv * stepsize


def cov(a):
    # return np.cov(a,rowvar=0,bias=1)
    mu = a.mean(0)
    if isinstance(a,np.ma.MaskedArray):
        return np.ma.dot(a.T,a)/a.count(0)[0] - np.ma.outer(mu,mu)
    else:
        return a.T.dot(a)/a.shape[0] - np.outer(mu,mu)

def normal_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return 0.5 * special.erfc(-z / np.sqrt(2))


### Sampling functions

def sample_gaussian(mu=None,Sigma=None,J=None,h=None):
    mean_params = mu is not None and Sigma is not None
    info_params = J is not None and h is not None
    assert mean_params or info_params

    if not any_none(mu,Sigma):
        return np.random.multivariate_normal(mu,Sigma)
    else:
        from scipy.linalg.lapack import dpotrs
        L = np.linalg.cholesky(J)
        x = np.random.randn(h.shape[0])
        return scipy.linalg.solve_triangular(L,x,lower=True,trans='T') \
            + dpotrs(L,h,lower=True)[0]

def sample_truncated_gaussian(mu=0, sigma=1, lb=-np.Inf, ub=np.Inf):
    """
    Sample a truncated normal with the specified params. This
    is not the most stable way but it works as long as the
    truncation region is not too far from the mean.
    """
    # Broadcast arrays to be of the same shape
    mu, sigma, lb, ub = np.broadcast_arrays(mu, sigma, lb, ub)
    shp = mu.shape
    if np.allclose(sigma, 0.0):
        return mu

    cdflb = normal_cdf(lb, mu, sigma)
    cdfub = normal_cdf(ub, mu, sigma)

    # Sample uniformly from the CDF
    cdfsamples = cdflb + np.random.rand(*shp) * (cdfub-cdflb)

    # Clip the CDF samples so that we can invert them
    cdfsamples = np.clip(cdfsamples, 1e-15, 1-1e-15)
    zs = -np.sqrt(2) * special.erfcinv(2 * cdfsamples)

    # Transform the standard normal samples
    xs = sigma * zs + mu
    xs = np.clip(xs, lb, ub)

    return xs

def sample_discrete(distn,size=[],dtype=np.int32):
    'samples from a one-dimensional finite pmf'
    distn = np.atleast_1d(distn)
    assert (distn >=0).all() and distn.ndim == 1
    if (0 == distn).all():
        return np.random.randint(distn.shape[0],size=size)
    cumvals = np.cumsum(distn)
    return np.sum(np.array(random(size))[...,na] * cumvals[-1] > cumvals, axis=-1,dtype=dtype)

def sample_discrete_from_log(p_log,return_lognorms=False,axis=0,dtype=np.int32):
    'samples log probability array along specified axis'
    lognorms = logsumexp(p_log,axis=axis)
    cumvals = np.exp(p_log - np.expand_dims(lognorms,axis)).cumsum(axis)
    thesize = np.array(p_log.shape)
    thesize[axis] = 1
    randvals = random(size=thesize) * \
            np.reshape(cumvals[[slice(None) if i is not axis else -1
                for i in range(p_log.ndim)]],thesize)
    samples = np.sum(randvals > cumvals,axis=axis,dtype=dtype)
    if return_lognorms:
        return samples, lognorms
    else:
        return samples

def sample_markov(T,trans_matrix,init_state_distn):
    out = np.empty(T,dtype=np.int32)
    out[0] = sample_discrete(init_state_distn)
    for t in range(1,T):
        out[t] = sample_discrete(trans_matrix[out[t-1]])
    return out

def sample_invgamma(alpha, beta):
    return 1./np.random.gamma(alpha, 1./beta)

def niw_expectedstats(nu, S, m, kappa):
    D = m.shape[0]

    # TODO speed this up with cholesky of S
    E_J = nu * np.linalg.inv(S)
    E_h = nu * np.linalg.solve(S,m)
    E_muJmuT = D/kappa + m.dot(E_h)
    E_logdetSigmainv = special.digamma((nu-np.arange(D))/2.).sum() \
        + D*np.log(2.) - np.linalg.slogdet(S)[1]

    return E_J, E_h, E_muJmuT, E_logdetSigmainv


def sample_niw(mu,lmbda,kappa,nu):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.
    '''
    # code is based on Matlab's method
    # reference: p. 87 in Gelman's Bayesian Data Analysis
    assert nu > lmbda.shape[0] and kappa > 0

    # first sample Sigma ~ IW(lmbda,nu)
    lmbda = sample_invwishart(lmbda,nu)
    # then sample mu | Lambda ~ N(mu, Lambda/kappa)
    mu = np.random.multivariate_normal(mu,lmbda / kappa)

    return mu, lmbda

def sample_invwishart(S,nu):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(int(nu),n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)//2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)

def sample_wishart(sigma, nu):
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (nu <= 81+n) and (nu == round(nu)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,nu)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(nu - np.arange(n))))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)

def sample_mn(M, U=None, Uinv=None, V=None, Vinv=None):
    assert (U is None) ^ (Uinv is None)
    assert (V is None) ^ (Vinv is None)

    G = np.random.normal(size=M.shape)

    if U is not None:
        G = np.dot(np.linalg.cholesky(U),G)
    else:
        G = np.linalg.solve(np.linalg.cholesky(Uinv).T,G)

    if V is not None:
        G = np.dot(G,np.linalg.cholesky(V).T)
    else:
        G = np.linalg.solve(np.linalg.cholesky(Vinv).T,G.T).T

    return M + G

def sample_mniw(nu, S, M, K=None, Kinv=None):
    assert (K is None) ^ (Kinv is None)
    Sigma = sample_invwishart(S,nu)
    if K is not None:
        return sample_mn(M=M,U=Sigma,V=K), Sigma
    else:
        return sample_mn(M=M,U=Sigma,Vinv=Kinv), Sigma

def mniw_expectedstats(nu, S, M, K=None, Kinv=None):
    # NOTE: could speed this up with chol factorizing S, not re-solving
    assert (K is None) ^ (Kinv is None)
    m = M.shape[0]
    K = K if K is not None else np.linalg.inv(Kinv)

    E_Sigmainv = nu*np.linalg.inv(S)
    E_Sigmainv_A = nu*np.linalg.solve(S,M)
    E_AT_Sigmainv_A = m*K + nu*M.T.dot(np.linalg.solve(S,M))
    E_logdetSigmainv = special.digamma((nu-np.arange(m))/2.).sum() \
        + m*np.log(2) - np.linalg.slogdet(S)[1]

    return E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv

def mniw_log_partitionfunction(nu, S, M, K):
    n = M.shape[0]
    return n*nu/2*np.log(2) + special.multigammaln(nu/2., n) \
        - nu/2*np.linalg.slogdet(S)[1] - n/2*np.linalg.slogdet(K)[1]

def sample_pareto(x_m,alpha):
    return x_m + np.random.pareto(alpha)

def sample_crp_tablecounts(concentration,customers,colweights):
    m = np.zeros_like(customers)
    tot = customers.sum()
    randseq = np.random.random(tot)

    starts = np.empty_like(customers)
    starts[0,0] = 0
    starts.flat[1:] = np.cumsum(np.ravel(customers)[:customers.size-1])

    for (i,j), n in np.ndenumerate(customers):
        w = colweights[j]
        for k in range(n):
            m[i,j] += randseq[starts[i,j]+k] \
                    < (concentration * w) / (k + concentration * w)

    return m

### Entropy
def invwishart_entropy(sigma,nu,chol=None):
    D = sigma.shape[0]
    chol = np.linalg.cholesky(sigma) if chol is None else chol
    Elogdetlmbda = special.digamma((nu-np.arange(D))/2).sum() + D*np.log(2) - 2*np.log(chol.diagonal()).sum()
    return invwishart_log_partitionfunction(sigma,nu,chol)-(nu-D-1)/2*Elogdetlmbda + nu*D/2

def invwishart_log_partitionfunction(sigma,nu,chol=None):
    # In Bishop B.79 notation, this is -log B(W, nu), where W = sigma^{-1}
    D = sigma.shape[0]
    chol = np.linalg.cholesky(sigma) if chol is None else chol
    return -1*(nu*np.log(chol.diagonal()).sum() - (nu*D/2*np.log(2) + D*(D-1)/4*np.log(np.pi) \
            + special.gammaln((nu-np.arange(D))/2).sum()))

### Predictive

def multivariate_t_loglik(y,nu,mu,lmbda):
    # returns the log value
    d = len(mu)
    yc = np.array(y-mu,ndmin=2)
    L = np.linalg.cholesky(lmbda)
    ys = scipy.linalg.solve_triangular(L,yc.T,overwrite_b=True,lower=True)
    return scipy.special.gammaln((nu+d)/2.) - scipy.special.gammaln(nu/2.) \
            - (d/2.)*np.log(nu*np.pi) - np.log(L.diagonal()).sum() \
            - (nu+d)/2.*np.log1p(1./nu*inner1d(ys.T,ys.T))

def beta_predictive(priorcounts,newcounts):
    prior_nsuc, prior_nfail = priorcounts
    nsuc, nfail = newcounts

    numer = scipy.special.gammaln(np.array([nsuc+prior_nsuc,
        nfail+prior_nfail, prior_nsuc+prior_nfail])).sum()
    denom = scipy.special.gammaln(np.array([prior_nsuc, prior_nfail,
        prior_nsuc+prior_nfail+nsuc+nfail])).sum()
    return numer - denom

### Statistical tests

def two_sample_t_statistic(pop1, pop2):
    pop1, pop2 = (flattendata(p) for p in (pop1, pop2))
    t = (pop1.mean(0) - pop2.mean(0)) / np.sqrt(pop1.var(0)/pop1.shape[0] + pop2.var(0)/pop2.shape[0])
    p = 2*stats.t.sf(np.abs(t),np.minimum(pop1.shape[0],pop2.shape[0]))
    return t,p

def f_statistic(pop1, pop2): # TODO test
    pop1, pop2 = (flattendata(p) for p in (pop1, pop2))
    var1, var2 = pop1.var(0), pop2.var(0)
    n1, n2 = np.where(var1 >= var2, pop1.shape[0], pop2.shape[0]), \
             np.where(var1 >= var2, pop2.shape[0], pop1.shape[0])
    var1, var2 = np.maximum(var1,var2), np.minimum(var1,var2)
    f = var1 / var2
    p = stats.f.sf(f,n1,n2)
    return f,p

