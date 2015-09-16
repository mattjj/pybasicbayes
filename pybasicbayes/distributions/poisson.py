from __future__ import division
from builtins import zip
__all__ = ['Poisson']
import numpy as np
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.abstractions import GibbsSampling, Collapsed, \
    MaxLikelihood, MeanField, MeanFieldSVI


class Poisson(GibbsSampling, Collapsed, MaxLikelihood, MeanField, MeanFieldSVI):
    '''
    Poisson distribution with a conjugate Gamma prior.

    NOTE: the support is {0,1,2,...}

    Hyperparameters (following Wikipedia's notation):
        alpha_0, beta_0

    Parameter is the mean/variance parameter:
        lmbda
    '''
    def __init__(self,lmbda=None,alpha_0=None,beta_0=None,mf_alpha_0=None,mf_beta_0=None):
        self.lmbda = lmbda

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mf_alpha_0 = mf_alpha_0 if mf_alpha_0 is not None else alpha_0
        self.mf_beta_0 = mf_beta_0 if mf_beta_0 is not None else beta_0

        if lmbda is None and not any(_ is None for _ in (alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def params(self):
        return dict(lmbda=self.lmbda)

    @property
    def hypparams(self):
        return dict(alpha_0=self.alpha_0,beta_0=self.beta_0)

    def log_sf(self,x):
        return stats.poisson.logsf(x,self.lmbda)

    def _posterior_hypparams(self,n,tot):
        return self.alpha_0 + tot, self.beta_0 + n

    def rvs(self,size=None):
        return np.random.poisson(self.lmbda,size=size)

    def log_likelihood(self,x):
        lmbda = self.lmbda
        x = np.array(x,ndmin=1)
        raw = np.empty(x.shape)
        raw[x>=0] = -lmbda + x[x>=0]*np.log(lmbda) - special.gammaln(x[x>=0]+1)
        raw[x<0] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum()
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert np.isscalar(data)
            n = 1
            tot = data

        return n, tot

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            n = weights.sum()
            tot = weights.dot(data)
        elif isinstance(data,list):
            n = sum(w.sum() for w in weights)
            tot = sum(w.dot(d) for w,d in zip(weights,data))
        else:
            assert np.isscalar(data) and np.isscalar(weights)
            n = weights
            tot = weights*data

        return np.array([n, tot])

    ### Gibbs Sampling

    def resample(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        alpha_n, beta_n = self._posterior_hypparams(*stats)
        self.lmbda = np.random.gamma(alpha_n,1/beta_n)

        # next line is for mean field initialization
        self.mf_alpha_0, self.mf_beta_0 = self.lmbda * self.beta_0, self.beta_0

        return self

    ### Mean Field

    def _resample_from_mf(self):
        mf_alpha_0, mf_beta_0 = self._natural_to_standard(self.mf_natural_hypparam)
        self.lmbda = np.random.gamma(mf_alpha_0, 1./mf_beta_0)

    def meanfieldupdate(self,data,weights):
        self.mf_natural_hypparam = \
                self.natural_hypparam + self._get_weighted_statistics(data,weights)
        self.lmbda = self.mf_alpha_0 / self.mf_beta_0

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        self.mf_natural_hypparam = \
                (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                        self.natural_hypparam
                        + 1./prob * self._get_weighted_statistics(data,weights))

    def get_vlb(self):
        return (self.natural_hypparam - self.mf_natural_hypparam).dot(self._mf_expected_statistics) \
                - (self._log_partition_fn(self.alpha_0,self.beta_0)
                        - self._log_partition_fn(self.mf_alpha_0,self.mf_beta_0))

    def expected_log_likelihood(self,x):
        Emlmbda, Elnlmbda = self._mf_expected_statistics
        return -special.gammaln(x+1) + Elnlmbda * x + Emlmbda

    @property
    def _mf_expected_statistics(self):
        alpha, beta = self.mf_alpha_0, self.mf_beta_0
        return np.array([-alpha/beta, special.digamma(alpha) - np.log(beta)])


    @property
    def natural_hypparam(self):
        return self._standard_to_natural(self.alpha_0,self.beta_0)

    @property
    def mf_natural_hypparam(self):
        return self._standard_to_natural(self.mf_alpha_0,self.mf_beta_0)

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.mf_alpha_0, self.mf_beta_0 = self._natural_to_standard(natparam)


    def _standard_to_natural(self,alpha,beta):
        return np.array([beta, alpha-1])

    def _natural_to_standard(self,natparam):
        return natparam[1]+1, natparam[0]

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_fn(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_fn(self.alpha_0,self.beta_0) \
                - self._get_sum_of_gammas(data)

    def _log_partition_fn(self,alpha,beta):
        return special.gammaln(alpha) - alpha * np.log(beta)

    def _get_sum_of_gammas(self,data):
        if isinstance(data,np.ndarray):
            return special.gammaln(data+1).sum()
        elif isinstance(data,list):
            return sum(special.gammaln(d+1).sum() for d in data)
        else:
            assert isinstance(data,int)
            return special.gammaln(data+1)

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        if n > 1e-2:
            self.lmbda = tot/n
            assert self.lmbda > 0
        else:
            self.broken = True
            self.lmbda = 999999

        return self

