from __future__ import division
from builtins import zip
from builtins import range
__all__ = ['Regression', 'RegressionNonconj', 'ARDRegression',
           'AutoRegression', 'ARDAutoRegression', 'DiagonalRegression']

import numpy as np
from numpy import newaxis as na

from pybasicbayes.abstractions import GibbsSampling, MaxLikelihood, \
    MeanField, MeanFieldSVI
from pybasicbayes.util.stats import sample_gaussian, sample_mniw, \
    sample_invwishart, getdatasize, mniw_expectedstats, mniw_log_partitionfunction, \
    sample_invgamma, update_param

from pybasicbayes.util.general import blockarray, inv_psd, cumsum, \
    all_none, any_none, AR_striding, objarray


class Regression(GibbsSampling, MeanField, MaxLikelihood):
    def __init__(
            self, nu_0=None,S_0=None,M_0=None,K_0=None,
            affine=False,
            A=None,sigma=None):
        self.affine = affine

        self._check_shapes(A, sigma, nu_0, S_0, M_0, K_0)

        self.A = A
        self.sigma = sigma

        have_hypers = not any_none(nu_0,S_0,M_0,K_0)

        if have_hypers:
            self.natural_hypparam = self.mf_natural_hypparam = \
                self._standard_to_natural(nu_0,S_0,M_0,K_0)

        if A is sigma is None and have_hypers:
            self.resample()  # initialize from prior

    @staticmethod
    def _check_shapes(A, sigma, nu, S, M, K):
        is_2d = lambda x: isinstance(x, np.ndarray) and x.ndim == 2
        not_none = lambda x: x is not None
        assert all(map(is_2d, filter(not_none, [A, sigma, S, M, K]))), 'Matrices must be 2D'

        get_dim = lambda x, i: x.shape[i] if x is not None else None
        get_dim_list = lambda pairs: filter(not_none, map(get_dim, *zip(*pairs)))
        is_consistent = lambda dimlist: len(set(dimlist)) == 1
        dims_agree = lambda pairs: is_consistent(get_dim_list(pairs))
        assert dims_agree([(A, 1), (M, 1), (K, 0), (K, 1)]), 'Input dimensions not consistent'
        assert dims_agree([(A, 0), (sigma, 0), (sigma, 1), (S, 0), (S, 1), (M, 0)]), \
            'Output dimensions not consistent'

    @property
    def parameters(self):
        return (self.A, self.sigma)

    @parameters.setter
    def parameters(self, A_sigma_tuple):
        (A,sigma) = A_sigma_tuple
        self.A = A
        self.sigma = sigma

    @property
    def D_in(self):
        # NOTE: D_in includes the extra affine coordinate
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[1]

    @property
    def D_out(self):
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[0]

    ### converting between natural and standard parameters

    @staticmethod
    def _standard_to_natural(nu,S,M,K):
        Kinv = inv_psd(K)
        A = S + M.dot(Kinv).dot(M.T)
        B = M.dot(Kinv)
        C = Kinv
        d = nu
        return np.array([A,B,C,d])

    @staticmethod
    def _natural_to_standard(natparam):
        A,B,C,d = natparam
        nu = d
        Kinv = C
        K = inv_psd(Kinv)
        M = B.dot(K)
        S = A - B.dot(K).dot(B.T)

        # numerical padding here...
        K += 1e-8*np.eye(K.shape[0])
        S += 1e-8*np.eye(S.shape[0])
        assert np.all(0 < np.linalg.eigvalsh(S))
        assert np.all(0 < np.linalg.eigvalsh(K))

        return nu, S, M, K

    ### getting statistics

    # NOTE: stats object arrays depend on the last element being a scalar,
    # otherwise numpy will try to create a dense array and fail

    def _get_statistics(self,data):
        assert isinstance(data, (list, tuple, np.ndarray))
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),
                       self._empty_statistics())
        elif isinstance(data, tuple):
            x, y = data
            bad = np.isnan(x).any(1) | np.isnan(y).any(1)
            x, y = x[~bad], y[~bad]

            n, D = y.shape

            xxT, yxT, yyT = \
                x.T.dot(x), y.T.dot(x), y.T.dot(y)

            if self.affine:
                x, y = x.sum(0), y.sum(0)
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])
        else:
            # data passed in like np.hstack((x, y))
            data = data[~np.isnan(data).any(1)]
            n, D = data.shape[0], self.D_out

            statmat = data.T.dot(data)
            xxT, yxT, yyT = \
                statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = data.sum(0)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])

    def _get_weighted_statistics(self,data,weights):
        assert isinstance(data, (list, tuple, np.ndarray))
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),
                       self._empty_statistics())
        elif isinstance(data, tuple):
            x, y = data
            bad = np.isnan(x).any(1) | np.isnan(y).any(1)
            x, y, weights = x[~bad], y[~bad], weights[~bad]

            n, D = weights.sum(), y.shape[1]
            wx = weights[:,na]*x

            xxT, yxT, yyT = \
                x.T.dot(wx), y.T.dot(wx), y.T.dot(weights[:,na]*y)

            if self.affine:
                x, y = weights.dot(x), weights.dot(y)
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])
        else:
            # data passed in like np.hstack((x, y))
            gi = ~np.isnan(data).any(1)
            data, weights = data[gi], weights[gi]
            n, D = weights.sum(), self.D_out

            statmat = data.T.dot(weights[:,na]*data)
            xxT, yxT, yyT = \
                statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = weights.dot(data)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])

    def _empty_statistics(self):
        D_in, D_out = self.D_in, self.D_out
        return np.array(
            [np.zeros((D_out,D_out)), np.zeros((D_out,D_in)),
             np.zeros((D_in,D_in)),0])

    @staticmethod
    def _stats_ensure_array(stats):
        if isinstance(stats, np.ndarray):
            return stats
        affine = len(stats) > 4

        yyT, yxT, xxT, n = stats[-4:]
        if affine:
            y, x = stats[:2]
            yxT = np.hstack((yxT, y[:,None]))
            xxT = blockarray([[xxT, x[:,None]], [x[None,:], 1.]])

        return np.array([yyT, yxT, xxT, n])

    ### distribution

    def log_likelihood(self,xy):
        assert isinstance(xy, (tuple,np.ndarray))
        A, sigma, D = self.A, self.sigma, self.D_out
        x, y = (xy[:,:-D], xy[:,-D:]) if isinstance(xy,np.ndarray) else xy

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        sigma_inv, L = inv_psd(sigma, return_chol=True)
        parammat = -1./2 * blockarray([
            [A.T.dot(sigma_inv).dot(A), -A.T.dot(sigma_inv)],
            [-sigma_inv.dot(A), sigma_inv]])

        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        if isinstance(xy, np.ndarray):
            out = np.einsum(contract,xy.dot(parammat),xy)
        else:
            out = np.einsum(contract,x.dot(parammat[:-D,:-D]),x)
            out += np.einsum(contract,y.dot(parammat[-D:,-D:]),y)
            out += 2*np.einsum(contract,x.dot(parammat[:-D,-D:]),y)

        out -= D/2*np.log(2*np.pi) + np.log(np.diag(L)).sum()

        if self.affine:
            out += y.dot(sigma_inv).dot(b)
            out -= x.dot(A.T).dot(sigma_inv).dot(b)
            out -= 1./2*b.dot(sigma_inv).dot(b)

        return out

    def predict(self, x):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:, :-1], A[:, -1]
            y = x.dot(A.T) + b.T
        else:
            y = x.dot(A.T)

        return y

    def rvs(self,x=None,size=1,return_xy=True):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        x = np.random.normal(size=(size,A.shape[1])) if x is None else x
        y = self.predict(x)
        y += np.random.normal(size=(x.shape[0], self.D_out)) \
            .dot(np.linalg.cholesky(sigma).T)

        return np.hstack((x,y)) if return_xy else y

    ### Gibbs sampling

    def resample(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        self.A, self.sigma = sample_mniw(
            *self._natural_to_standard(self.natural_hypparam + stats))
        self._initialize_mean_field()

    ### Max likelihood

    def max_likelihood(self,data,weights=None,stats=None):
        if stats is None:
            stats = self._get_statistics(data) if weights is None \
                else self._get_weighted_statistics(data,weights)

        yyT, yxT, xxT, n = stats

        if n > 0:
            try:
                self.A = np.linalg.solve(xxT, yxT.T).T
                self.sigma = (yyT - self.A.dot(yxT.T))/n

                def symmetrize(A):
                    return (A + A.T)/2.
                self.sigma = 1e-10*np.eye(self.D_out) \
                    + symmetrize(self.sigma)  # numerical
            except np.linalg.LinAlgError:
                self.broken = True
        else:
            self.broken = True

        assert np.allclose(self.sigma,self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        self._initialize_mean_field()

        return self

    ### Mean Field

    def meanfieldupdate(self, data=None, weights=None, stats=None):
        assert (data is not None and weights is not None) ^ (stats is not None)
        stats = self._stats_ensure_array(stats) if stats is not None \
            else self._get_weighted_statistics(data, weights)
        self.mf_natural_hypparam = self.natural_hypparam + stats
        self._set_params_from_mf()

    def meanfield_sgdstep(self, data, weights, prob, stepsize, stats=None):
        if stats is None:
            stats = self._get_weighted_statistics(data, weights)
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize \
            * (self.natural_hypparam + 1./prob * stats)
        self._set_params_from_mf()

    def meanfield_expectedstats(self):
        from pybasicbayes.util.stats import mniw_expectedstats
        return mniw_expectedstats(
                *self._natural_to_standard(self.mf_natural_hypparam))

    def expected_log_likelihood(self, xy=None, stats=None):
        # TODO test values, test for the affine case
        assert isinstance(xy, (tuple, np.ndarray)) ^ isinstance(stats, tuple)

        D = self.D_out
        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = \
            mniw_expectedstats(
                *self._natural_to_standard(self.mf_natural_hypparam))

        if self.affine:
            E_Sigmainv_A, E_Sigmainv_b = \
                E_Sigmainv_A[:,:-1], E_Sigmainv_A[:,-1]
            E_AT_Sigmainv_A, E_AT_Sigmainv_b, E_bT_Sigmainv_b = \
                E_AT_Sigmainv_A[:-1,:-1], E_AT_Sigmainv_A[:-1,-1], \
                E_AT_Sigmainv_A[-1,-1]

        if xy is not None:
            x, y = (xy[:,:-D], xy[:,-D:]) if isinstance(xy, np.ndarray) \
                else xy

            parammat = -1./2 * blockarray([
                [E_AT_Sigmainv_A, -E_Sigmainv_A.T],
                [-E_Sigmainv_A, E_Sigmainv]])

            contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
            if isinstance(xy, np.ndarray):
                out = np.einsum('ni,ni->n', xy.dot(parammat), xy)
            else:
                out = np.einsum(contract,x.dot(parammat[:-D,:-D]),x)
                out += np.einsum(contract,y.dot(parammat[-D:,-D:]),y)
                out += 2*np.einsum(contract,x.dot(parammat[:-D,-D:]),y)

            out += -D/2*np.log(2*np.pi) + 1./2*E_logdetSigmainv

            if self.affine:
                out += y.dot(E_Sigmainv_b)
                out -= x.dot(E_AT_Sigmainv_b)
                out -= 1./2 * E_bT_Sigmainv_b
        else:
            if self.affine:
                Ey, Ex = stats[:2]
            yyT, yxT, xxT, n = stats[-4:]

            contract = 'ij,nij->n' if yyT.ndim == 3 else 'ij,ij->'

            out = -1./2 * np.einsum(contract, E_AT_Sigmainv_A, xxT)
            out += np.einsum(contract, E_Sigmainv_A, yxT)
            out += -1./2 * np.einsum(contract, E_Sigmainv, yyT)
            out += -D/2*np.log(2*np.pi) + n/2.*E_logdetSigmainv

            if self.affine:
                out += Ey.dot(E_Sigmainv_b)
                out -= Ex.dot(E_AT_Sigmainv_b)
                out -= 1./2 * E_bT_Sigmainv_b

        return out

    def get_vlb(self):
        E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv = \
            mniw_expectedstats(*self._natural_to_standard(self.mf_natural_hypparam))
        A, B, C, d = self.natural_hypparam - self.mf_natural_hypparam
        bilinear_term = -1./2 * np.trace(A.dot(E_Sigmainv)) \
            + np.trace(B.T.dot(E_Sigmainv_A)) \
            - 1./2 * np.trace(C.dot(E_AT_Sigmainv_A)) \
            + 1./2 * d * E_logdetSigmainv

        # log normalizer term
        Z = mniw_log_partitionfunction(*self._natural_to_standard(
            self.natural_hypparam))
        Z_mf = mniw_log_partitionfunction(*self._natural_to_standard(
            self.mf_natural_hypparam))

        return bilinear_term - (Z - Z_mf)

    def resample_from_mf(self):
        self.A, self.sigma = sample_mniw(
            *self._natural_to_standard(self.mf_natural_hypparam))

    def _set_params_from_mf(self):
        nu, S, M, K = self._natural_to_standard(self.mf_natural_hypparam)
        self.A, self.sigma = M, S / nu

    def _initialize_mean_field(self):
        if hasattr(self, 'natural_hypparam'):
            A, Sigma = self.A, self.sigma
            nu, S, M, K = self._natural_to_standard(self.natural_hypparam)
            self.mf_natural_hypparam = self._standard_to_natural(
                nu, nu*Sigma, A, K)


class RegressionNonconj(Regression):
    def __init__(self, M_0, Sigma_0, nu_0, S_0,
                 A=None, sigma=None, affine=False, niter=10):
        self.A = A
        self.sigma = sigma
        self.affine = affine

        self.h_0 = np.linalg.solve(Sigma_0, M_0.ravel()).reshape(M_0.shape)
        self.J_0 = np.linalg.inv(Sigma_0)
        self.nu_0 = nu_0
        self.S_0 = S_0

        self.niter = niter

        if all_none(A,sigma):
            self.resample()  # initialize from prior

    ### Gibbs

    def resample(self,data=[],niter=None):
        niter = niter if niter else self.niter
        if getdatasize(data) == 0:
            self.A = sample_gaussian(J=self.J_0,h=self.h_0.ravel())\
                .reshape(self.h_0.shape)
            self.sigma = sample_invwishart(self.S_0,self.nu_0)
        else:
            yyT, yxT, xxT, n = self._get_statistics(data)
            for itr in range(niter):
                self._resample_A(xxT, yxT, self.sigma)
                self._resample_sigma(xxT, yxT, yyT, n, self.A)

    def _resample_A(self, xxT, yxT, sigma):
        sigmainv = np.linalg.inv(sigma)
        J = self.J_0 + np.kron(sigmainv, xxT)
        h = self.h_0 + sigmainv.dot(yxT)
        self.A = sample_gaussian(J=J,h=h.ravel()).reshape(h.shape)

    def _resample_sigma(self, xxT, yxT, yyT, n, A):
        S = self.S_0 + yyT - yxT.dot(A.T) - A.dot(yxT.T) + A.dot(xxT).dot(A.T)
        nu = self.nu_0 + n
        self.sigma = sample_invwishart(S, nu)


class ARDRegression(Regression):
    def __init__(
            self, a,b,nu_0,S_0,M_0,
            blocksizes=None,K_0=None,niter=10,**kwargs):
        blocksizes = np.ones(M_0.shape[1],dtype=np.int64) \
            if blocksizes is None else blocksizes
        self.niter = niter
        self.blocksizes = np.array(blocksizes)
        self.starts = cumsum(blocksizes,strict=True)
        self.stops = cumsum(blocksizes,strict=False)

        self.a = np.repeat(a,len(blocksizes))
        self.b = np.repeat(b,len(blocksizes))

        self.nu_0 = nu_0
        self.S_0 = S_0
        self.M_0 = M_0

        if K_0 is None:
            self.resample_K()
        else:
            self.K_0 = K_0

        super(ARDRegression,self).__init__(
            K_0=self.K_0,nu_0=nu_0,S_0=S_0,M_0=M_0,**kwargs)

    def resample(self,data=[],stats=None):
        if len(data) > 0 or stats is not None:
            stats = self._get_statistics(data) if stats is None else stats
            for itr in range(self.niter):
                self.A, self.sigma = \
                    sample_mniw(*self._natural_to_standard(
                        self.natural_hypparam + stats))

                mat = self.M_0 - self.A
                self.resample_K(1./2*np.einsum(
                    'ij,ij->j',mat,np.linalg.solve(self.sigma,mat)))
        else:
            self.resample_K()
            super(ARDRegression,self).resample()

    def resample_K(self,diag=None):
        if diag is None:
            a, b = self.a, self.b
        else:
            sums = [diag[start:stop].sum()
                    for start,stop in zip(self.starts,self.stops)]
            a = self.a + self.D_out*self.blocksizes/2.
            b = self.b + np.array(sums)

        ks = 1./np.random.gamma(a,scale=1./b)
        self.K_0 = np.diag(np.repeat(ks,self.blocksizes))

        self.natural_hypparam = self._standard_to_natural(
            self.nu_0,self.S_0,self.M_0,self.K_0)

    @property
    def parameters(self):
        return (self.A, self.sigma, self.K_0)

    @parameters.setter
    def parameters(self, A_sigma_K_0_tuple1):
        (A,sigma,K_0) = A_sigma_K_0_tuple1
        self.A = A
        self.sigma = sigma
        self.K_0 = K_0


class DiagonalRegression(Regression, MeanFieldSVI):
    """
    Special case of the regression class in which the observations
    have diagonal Gaussian noise and, potentially, missing data.
    """

    def __init__(self, D_out, D_in, mu_0=None, Sigma_0=None, alpha_0=3.0, beta_0=2.0,
                 A=None, sigmasq=None, niter=1):

        self._D_out = D_out
        self._D_in = D_in
        self.A = A
        self.sigmasq_flat = sigmasq
        self.affine = False # We do not yet support affine

        mu_0 = np.zeros(D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(D_in) if Sigma_0 is None else Sigma_0
        assert mu_0.shape == (D_in,)
        assert Sigma_0.shape == (D_in, D_in)
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.niter = niter

        if any_none(A, sigmasq):
            self.A = np.zeros((D_out, D_in))
            self.sigmasq_flat = np.ones((D_out,))
            self.resample(data=None)  # initialize from prior

        # Store the natural parameters and expose the standard versions as properties
        self.mf_J_A = np.array([self.J_0.copy() for _ in range(D_out)])
        # Initializing with mean zero is pathological. Break symmetry by starting with sampled A.
        # self.mf_h_A = np.array([self.h_0.copy() for _ in range(D_out)])
        self.mf_h_A = np.array([Jd.dot(Ad) for Jd,Ad in zip(self.mf_J_A, self.A)])

        self.mf_alpha = self.alpha_0 * np.ones(D_out)
        self.mf_beta = self.alpha_0 * self.sigmasq_flat

        # Cache the standard parameters for A as well
        self._mf_A_cache = {}

    @property
    def D_out(self):
        return self._D_out

    @property
    def D_in(self):
        return self._D_in

    @property
    def sigma(self):
        return np.diag(self.sigmasq_flat)

    @property
    def mf_expectations(self):
        # Look for expectations in the cache
        if ("mf_E_A" not in self._mf_A_cache) or \
            ("mf_E_AAT" not in self._mf_A_cache):
            mf_Sigma_A = \
                np.array([np.linalg.inv(Jd) for Jd in self.mf_J_A])

            self._mf_A_cache["mf_E_A"] = \
                np.array([np.dot(Sd, hd)
                          for Sd,hd in zip(mf_Sigma_A, self.mf_h_A)])

            self._mf_A_cache["mf_E_AAT"] = \
                np.array([Sd + np.outer(md,md)
                          for Sd,md in zip(mf_Sigma_A, self._mf_A_cache["mf_E_A"])])

        mf_E_A = self._mf_A_cache["mf_E_A"]
        mf_E_AAT = self._mf_A_cache["mf_E_AAT"]

        # Set the invgamma meanfield expectation
        from scipy.special import digamma
        mf_E_sigmasq_inv = self.mf_alpha / self.mf_beta
        mf_E_log_sigmasq = np.log(self.mf_beta) - digamma(self.mf_alpha)

        return mf_E_A, mf_E_AAT, mf_E_sigmasq_inv, mf_E_log_sigmasq

    # TODO: This is a bit ugly... Return stats in the form expected by PyLDS
    def meanfield_expectedstats(self):
        mf_E_A, mf_E_AAT, mf_E_sigmasq_inv, mf_E_log_sigmasq = self.mf_expectations
        E_Sigmainv = np.diag(mf_E_sigmasq_inv)
        E_Sigmainv_A  = mf_E_A * mf_E_sigmasq_inv[:,None]
        E_AT_Sigmainv_A = np.sum(mf_E_sigmasq_inv[:,None,None] * mf_E_AAT, axis=0)
        E_logdetSigmainv = -np.sum(mf_E_log_sigmasq)
        return E_Sigmainv, E_Sigmainv_A, E_AT_Sigmainv_A, E_logdetSigmainv

    def log_likelihood(self, xy, mask=None):
        if isinstance(xy, tuple):
            x,y = xy
        else:
            x,y = xy[:,:self.D_in], xy[:,self.D_in:]
            assert y.shape[1] == self.D_out

        if mask is None:
            mask = np.ones_like(y)
        else:
            assert mask.shape == y.shape

        sqerr = -0.5 * (y-x.dot(self.A.T))**2 * mask
        ll = np.sum(sqerr / self.sigmasq_flat, axis=1)

        # Add normalizer
        ll += np.sum(-0.5*np.log(2*np.pi*self.sigmasq_flat) * mask, axis=1)

        return ll

    def _get_statistics(self, data, D_out=None, D_in=None, mask=None):
        D_out = self.D_out if D_out is None else D_out
        D_in = self.D_in if D_in is None else D_in
        if data is None:
            return (np.zeros((D_out,)),
                    np.zeros((D_out, D_in)),
                    np.zeros((D_out, D_in, D_in)),
                    np.zeros((D_out,)))

        # Make sure data is a list
        if not isinstance(data, list):
            datas = [data]
        else:
            datas = data

        # Make sure mask is also a list if given
        if mask is not None:
            if not isinstance(mask, list):
                masks = [mask]
            else:
                masks = mask
        else:
            masks = [None] * len(datas)

        # Sum sufficient statistics from each dataset
        ysq = np.zeros(D_out)
        yxT = np.zeros((D_out, D_in))
        xxT = np.zeros((D_out, D_in, D_in))
        n = np.zeros(D_out)

        for data, mask in zip(datas, masks):
            # Dandle tuples or hstack-ed arrays
            if isinstance(data, tuple):
                x, y = data
            else:
                x, y = data[:,:D_in], data[:, D_in:]
            assert x.shape[1] == D_in
            assert y.shape[1] == D_out

            if mask is None:
                mask = np.ones_like(y, dtype=bool)

            ysq += np.sum(y**2 * mask, axis=0)
            yxT += (y*mask).T.dot(x)
            xxT += np.array([(x * mask[:,d][:,None]).T.dot(x)
                            for d in range(D_out)])
            n += np.sum(mask, axis=0)
        return ysq, yxT, xxT, n

    @staticmethod
    def _stats_ensure_array(stats):
        ysq, yxT, xxT, n = stats

        if yxT.ndim != 2:
            raise Exception("yxT.shape must be (D_out, D_in)")
        D_out, D_in = yxT.shape

        # If ysq is D_out x D_out, just take the diagonal
        if ysq.ndim == 1:
            assert ysq.shape == (D_out,)
        elif ysq.ndim == 2:
            assert ysq.shape == (D_out, D_out)
            ysq = np.diag(ysq)
        else:
            raise Exception("ysq.shape must be (D_out,) or (D_out, D_out)")

        # Make sure xxT is D_out x D_in x D_in
        if xxT.ndim == 2:
            assert xxT.shape == (D_in, D_in)
            xxT = np.tile(xxT[None,:,:], (D_out, 1, 1))
        elif xxT.ndim == 3:
            assert xxT.shape == (D_out, D_in, D_in)
        else:
            raise Exception("xxT.shape must be (D_in, D_in) or (D_out, D_in, D_in)")

        # Make sure n is of shape (D_out,)
        if np.isscalar(n):
            n = n * np.ones(D_out)
        elif n.ndim == 1:
            assert n.shape == (D_out,)
        else:
            raise Exception("n must be a scalar or an array of shape (D_out,)")

        return objarray([ysq, yxT, xxT, n])

    ### Gibbs
    def resample(self, data, stats=None, mask=None, niter=None):
        """
        Introduce a mask that allows for missing data
        """
        stats = self._get_statistics(data, mask=mask) if stats is None else stats
        stats = self._stats_ensure_array(stats)

        niter = niter if niter else self.niter
        for itr in range(niter):
            self._resample_A(stats)
            self._resample_sigma(stats)

    def _resample_A(self, stats):

        _, yxT, xxT, _ = stats

        # Sample each row of W
        for d in range(self.D_out):
            # Get sufficient statistics from the data
            Jd = self.J_0 + xxT[d] / self.sigmasq_flat[d]
            hd = self.h_0 + yxT[d] / self.sigmasq_flat[d]
            self.A[d] = sample_gaussian(J=Jd, h=hd)

    def _resample_sigma(self, stats):
        ysq, yxT, xxT, n = stats
        AAT = np.array([np.outer(a,a) for a in self.A])

        alpha = self.alpha_0 + n / 2.0

        beta = self.beta_0
        beta += 0.5 * ysq
        beta += -1.0 * np.sum(yxT * self.A, axis=1)
        beta += 0.5 * np.sum(AAT * xxT, axis=(1,2))

        self.sigmasq_flat = np.reshape(sample_invgamma(alpha, beta), (self.D_out,))

    ### Max likelihood
    def max_likelihood(self,data, weights=None, stats=None, mask=None):
        if stats is None:
            stats = self._get_statistics(data, mask)
        stats = self._stats_ensure_array(stats)

        ysq, yxT, xxT, n = stats

        assert np.all(n > 0), "Cannot perform max likelihood with zero data points!"
        self.A = np.array([
            np.linalg.solve(self.J_0 + xxTd, self.h_0 + yxTd)
            for xxTd, yxTd in zip(xxT, yxT)
        ])

        alpha = self.alpha_0 + n / 2.0
        beta = self.beta_0
        beta += 0.5 * ysq
        beta += -1.0 * np.sum(yxT * self.A, axis=1)
        AAT = np.array([np.outer(ad, ad) for ad in self.A])
        beta += 0.5 * np.sum(AAT * xxT, axis=(1, 2))

        self.sigmasq_flat = beta / (alpha + 1.0)
        assert np.all(self.sigmasq_flat >= 0)

    ### Mean Field
    def meanfieldupdate(self, data=None, weights=None, stats=None, mask=None):
        assert weights is None, "Not supporting weighted data, just masked data."
        if stats is None:
            stats = self._get_statistics(data, mask)
        stats = self._stats_ensure_array(stats)

        self._meanfieldupdate_A(stats)
        self._meanfieldupdate_sigma(stats)

    def _meanfieldupdate_A(self, stats, prob=1.0, stepsize=1.0):
        E_sigmasq_inv = self.mf_alpha / self.mf_beta
        _, E_yxT, E_xxT, _ = stats  / prob

        # Update statistics each row of A
        for d in range(self.D_out):
            Jd = self.J_0 + (E_xxT[d] * E_sigmasq_inv[d])
            hd = self.h_0 + (E_yxT[d] * E_sigmasq_inv[d])

            # Update the mean field natural parameters
            self.mf_J_A[d] = update_param(self.mf_J_A[d], Jd, stepsize)
            self.mf_h_A[d] = update_param(self.mf_h_A[d], hd, stepsize)

        # Clear the cache
        self._mf_A_cache = {}

    def _meanfieldupdate_sigma(self, stats, prob=1.0, stepsize=1.0):
        E_ysq, E_yxT, E_xxT, E_n = stats / prob
        E_A, E_AAT, _, _ = self.mf_expectations

        alpha = self.alpha_0 + E_n / 2.0

        beta = self.beta_0
        beta += 0.5 * E_ysq
        beta += -1.0 * np.sum(E_yxT * E_A, axis=1)
        beta += 0.5 * np.sum(E_AAT * E_xxT, axis=(1,2))

        # Set the invgamma meanfield parameters
        self.mf_alpha = update_param(self.mf_alpha, alpha, stepsize)
        self.mf_beta = update_param(self.mf_beta, beta, stepsize)

    def get_vlb(self):
        # TODO: Implement this!
        return 0


    def expected_log_likelihood(self, xy=None, stats=None, mask=None):
        if xy is not None:
            if isinstance(xy, tuple):
                x, y = xy
            else:
                x, y = xy[:, :self.D_in], xy[:, self.D_in:]
                assert y.shape[1] == self.D_out

            E_ysq = y**2
            E_yxT = y[:,:,None] * x[:,None,:]
            E_xxT = x[:,:,None] * x[:,None,:]
            E_n = np.ones_like(y) if mask is None else mask

        elif stats is not None:
            E_ysq, E_yxT, E_xxT, E_n = stats
            T = E_ysq.shape[0]
            assert E_ysq.shape == (T,self.D_out)
            assert E_yxT.shape == (T,self.D_out,self.D_in)

            if E_xxT.shape == (T, self.D_in, self.D_in):
                E_xxT = E_xxT[:, None, :, :]
            else:
                assert E_xxT.shape == (T,self.D_out,self.D_in,self.D_in)

            if E_n.shape == (T,):
                E_n = E_n[:,None]
            else:
                assert E_n.shape == (T,self.D_out)

        E_A, E_AAT, E_sigmasq_inv, E_log_sigmasq = self.mf_expectations

        sqerr = -0.5 * E_ysq
        sqerr += 1.0 * np.sum(E_yxT * E_A, axis=2)
        sqerr += -0.5 * np.sum(E_xxT * E_AAT, axis=(2,3))

        # Compute expected log likelihood
        ell = np.sum(sqerr * E_sigmasq_inv, axis=1)
        ell += np.sum(-0.5 * E_n * (E_log_sigmasq + np.log(2 * np.pi)), axis=1)

        return ell

    def resample_from_mf(self):
        for d in range(self.D_out):
            self.A[d] = sample_gaussian(J=self.mf_J_A[d], h=self.mf_h_A[d])
        self.sigmasq_flat = sample_invgamma(self.mf_alpha, self.mf_beta) * np.ones(self.D_out)

    def _initialize_mean_field(self):
        A, sigmasq = self.A, self.sigmasq_flat

        # Set mean field params such that A and sigmasq are the mean
        self.mf_alpha = self.alpha_0
        self.mf_beta = self.alpha_0 * sigmasq

        self.mf_J_A = np.array([self.J_0.copy() for _ in range(self.D_out)])
        self.mf_h_A = np.array([Jd.dot(Ad) for Jd, Ad in zip(self.mf_J_A, A)])

    ### SVI
    def meanfield_sgdstep(self, data, weights, prob, stepsize, stats=None, mask=None):
        assert weights is None, "Not supporting weighted datapoints (just masked data)"
        if stats is None:
            stats = self._get_statistics(data, mask)
        stats = self._stats_ensure_array(stats)

        self._meanfieldupdate_A(stats, prob=prob, stepsize=stepsize)
        self._meanfieldupdate_sigma(stats, prob=prob, stepsize=stepsize)


class _ARMixin(object):
    @property
    def nlags(self):
        if not self.affine:
            return self.D_in // self.D_out
        else:
            return (self.D_in - 1) // self.D_out

    @property
    def D(self):
        return self.D_out

    def predict(self, x):
        return super(_ARMixin,self).predict(np.atleast_2d(x.ravel()))

    def rvs(self,lagged_data):
        return super(_ARMixin,self).rvs(
                x=np.atleast_2d(lagged_data.ravel()),return_xy=False)

    def _get_statistics(self,data):
        return super(_ARMixin,self)._get_statistics(
                data=self._ensure_strided(data))

    def _get_weighted_statistics(self,data,weights):
        return super(_ARMixin,self)._get_weighted_statistics(
                data=self._ensure_strided(data),weights=weights)

    def log_likelihood(self,xy):
        return super(_ARMixin,self).log_likelihood(self._ensure_strided(xy))

    def _ensure_strided(self,data):
        if isinstance(data,np.ndarray):
            if data.shape[1] != self.D*(self.nlags+1):
                data = AR_striding(data,self.nlags)
            return data
        else:
            return [self._ensure_strided(d) for d in data]


class AutoRegression(_ARMixin,Regression):
    pass


class ARDAutoRegression(_ARMixin,ARDRegression):
    def __init__(self,M_0,**kwargs):
        blocksizes = [M_0.shape[0]]*(M_0.shape[1] // M_0.shape[0]) \
                + ([1] if M_0.shape[1] % M_0.shape[0] and M_0.shape[0] != 1 else [])
        super(ARDAutoRegression,self).__init__(
                M_0=M_0,blocksizes=blocksizes,**kwargs)
