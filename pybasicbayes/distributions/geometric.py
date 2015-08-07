from __future__ import division
from builtins import zip
__all__ = ['Geometric']

import numpy as np
import scipy.stats as stats
import scipy.special as special
from warnings import warn

from pybasicbayes.abstractions import GibbsSampling, MeanField, \
    Collapsed, MaxLikelihood


class Geometric(GibbsSampling, MeanField, Collapsed, MaxLikelihood):
    '''
    Geometric distribution with a conjugate beta prior.
    The support is {1,2,3,...}.

    Hyperparameters:
        alpha_0, beta_0

    Parameter is the success probability:
        p
    '''
    def __init__(self,alpha_0=None,beta_0=None,p=None):
        self.p = p

        self.alpha_0 = self.mf_alpha_0 = alpha_0
        self.beta_0 = self.mf_beta_0 = beta_0

        if p is None and not any(_ is None for _ in (alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def params(self):
        return dict(p=self.p)

    @property
    def hypparams(self):
        return dict(alpha_0=self.alpha_0,beta_0=self.beta_0)

    def _posterior_hypparams(self,n,tot):
        return self.alpha_0 + n, self.beta_0 + tot

    def log_likelihood(self,x):
        x = np.array(x,ndmin=1)
        raw = np.empty(x.shape)
        raw[x>0] = (x[x>0]-1.)*np.log(1.-self.p) + np.log(self.p)
        raw[x<1] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]

    def log_sf(self,x):
        return stats.geom.logsf(x,self.p)

    def pmf(self,x):
        return stats.geom.pmf(x,self.p)

    def rvs(self,size=None):
        return np.random.geometric(self.p,size=size)

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum() - n
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data) - n
        else:
            assert np.isscalar(data)
            n = 1
            tot = data-1
        return n, tot

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
             n = weights.sum()
             tot = weights.dot(data) - n
        elif isinstance(data,list):
            n = sum(w.sum() for w in weights)
            tot = sum(w.dot(d) for w,d in zip(weights,data)) - n
        else:
            assert np.isscalar(data) and np.isscalar(weights)
            n = weights
            tot = weights*data - 1

        return n, tot

    ### Gibbs sampling

    def resample(self,data=[]):
        self.p = np.random.beta(*self._posterior_hypparams(*self._get_statistics(data)))

        # initialize mean field
        self.alpha_mf = self.p*(self.alpha_0+self.beta_0)
        self.beta_mf = (1-self.p)*(self.alpha_0+self.beta_0)

        return self

    ### mean field

    def meanfieldupdate(self,data,weights,stats=None):
        warn('untested')
        n, tot = self._get_weighted_statistics(data,weights) if stats is None else stats
        self.alpha_mf = self.alpha_0 + n
        self.beta_mf = self.beta_0 + tot

        # initialize Gibbs
        self.p = self.alpha_mf / (self.alpha_mf + self.beta_mf)

    def get_vlb(self):
        warn('untested')
        Elnp, Eln1mp = self._expected_statistics(self.alpha_mf,self.beta_mf)
        return (self.alpha_0 - self.alpha_mf)*Elnp \
                + (self.beta_0 - self.beta_mf)*Eln1mp \
                - (self._log_partition_function(self.alpha_0,self.beta_0)
                        - self._log_partition_function(self.alpha_mf,self.beta_mf))

    def expected_log_likelihood(self,x):
        warn('untested')
        Elnp, Eln1mp = self._expected_statistics(self.alpha_mf,self.beta_mf)
        return (x-1)*Eln1mp + Elnp1mp

    def _expected_statistics(self,alpha,beta):
        warn('untested')
        Elnp = special.digamma(alpha) - special.digamma(alpha+beta)
        Eln1mp = special.digamma(beta) - special.digamma(alpha+beta)
        return Elnp, Eln1mp

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        self.p = n/tot
        return self

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
            - self._log_partition_function(self.alpha_0,self.beta_0)

    def _log_partition_function(self,alpha,beta):
        return special.betaln(alpha,beta)
