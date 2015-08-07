from __future__ import division
from builtins import zip
__all__ = ['Binomial']

import numpy as np
import scipy.stats as stats
import scipy.special as special
from warnings import warn

from pybasicbayes.abstractions import GibbsSampling, MeanField, \
    MeanFieldSVI


class Binomial(GibbsSampling, MeanField, MeanFieldSVI):
    '''
    Models a Binomial likelihood and a Beta prior:

        p ~ Beta(alpha_0, beta_0)
        x | p ~ Binom(p,n)

    where p is the success probability, alpha_0-1 is the prior number of
    successes, beta_0-1 is the prior number of failures.

    A special case of Multinomial where N is fixed and each observation counts
    the number of successes and is in {0,1,...,N}.
    '''
    def __init__(self,alpha_0,beta_0,alpha_mf=None,beta_mf=None,p=None,n=None):
        warn('this class is untested!')
        assert n is not None

        self.n = n
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.alpha_mf = alpha_mf if alpha_mf is not None else alpha_0
        self.beta_mf = beta_mf if beta_mf is not None else beta_0

        if p is not None:
            self.p = p
        else:
            self.resample()

    def log_likelihood(self,x):
        return stats.binom.pmf(x,self.n,self.p)

    def rvs(self,size=None):
        return stats.binom.pmf(self.n,self.p,size=size)

    @property
    def natural_hypparam(self):
        return np.array([self.alpha_0 - 1, self.beta_0 - 1])

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.alpha_0, self.beta_0 = natparam + 1

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            data = data.ravel()
            tot = data.sum()
            return np.array([tot, self.n*data.shape[0] - tot])
        else:
            return sum(
                (self._get_statistics(d) for d in data),
                self._empty_statistics())

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            data, weights = data.ravel(), weights.ravel()
            tot = weights.dot(data)
            return np.array([tot, self.n*weights.sum() - tot])
        else:
            return sum(
                (self._get_weighted_statistics(d,w) for d,w in zip(data,weights)),
                self._empty_statistics())

    def _empty_statistics(self):
        return np.zeros(2)

    ### Gibbs

    def resample(self,data=[]):
        alpha_n, beta_n = self.natural_hypparam + self._get_statistics(data) + 1
        self.p = np.random.beta(alpha_n,beta_n)

        # use Gibbs to initialize mean field
        self.alpha_mf = self.p * (self.alpha_0 + self.beta_0)
        self.beta_mf = (1-self.p) * (self.alpha_0 + self.beta_0)

    ### Mean field and SVI

    def meanfieldupdate(self,data,weights):
        self.mf_natural_hypparam = \
            self.natural_hypparam + self._get_weighted_statistics(data,weights)

        # use mean field to initialize Gibbs
        self.p = self.alpha_mf / (self.alpha_mf + self.beta_mf)

    def meanfield_sgdstep(self,data,weights,minibatchprob,stepsize):
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                self.natural_hypparam
                + 1./minibatchprob * self._get_weighted_statistics(data,weights))

    @property
    def mf_natural_hypparam(self):
        return np.array([self.alpha_mf - 1, self.beta_mf - 1])

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.alpha_mf, self.beta_mf = natparam + 1

    def expected_log_likelihood(self,x):
        n = self.n
        Elnp, Eln1mp = self._mf_expected_statistics()
        return special.gammaln(n+1) - special.gammaln(x+1) - special.gammaln(n-x+1) \
            + x*Elnp + (n-x)*Eln1mp

    def _mf_expected_statistics(self):
        return special.digamma([self.alpha_mf, self.beta_mf]) \
            - special.digamma(self.alpha_mf + self.beta_mf)

    def get_vlb(self):
        Elnp, Eln1mp = self._mf_expected_statistics()
        return (self.alpha_0 - self.alpha_mf)*Elnp \
            + (self.beta_0 - self.beta_mf)*Eln1mp \
            - (self._log_partition_function(self.alpha_0, self.beta_0)
                - self._log_partition_function(self.alpha_mf,self.beta_mf))

    def _log_partition_function(self,alpha,beta):
        return special.betaln(alpha,beta)
