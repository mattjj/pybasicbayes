from __future__ import division
from builtins import map
from builtins import range
__all__ = ['UniformOneSided', 'Uniform']

import numpy as np

from pybasicbayes.abstractions import GibbsSampling
from pybasicbayes.util.stats import sample_pareto
from pybasicbayes.util.general import any_none


class UniformOneSided(GibbsSampling):
    '''
    Models a uniform distribution over [low,high] for a parameter high.
    Low is a fixed hyperparameter (hence "OneSided"). See the Uniform class for
    the two-sided version.

    Likelihood is x ~ U[low,high]
    Prior is high ~ Pareto(x_m,alpha) following Wikipedia's notation

    Hyperparameters:
        x_m, alpha, low

    Parameters:
        high
    '''
    def __init__(self,high=None,x_m=None,alpha=None,low=0.):
        self.high = high

        self.x_m = x_m
        self.alpha = alpha
        self.low = low

        have_hypers = x_m is not None and alpha is not None
        if high is None and have_hypers:
            self.resample()  # intialize from prior

    @property
    def params(self):
        return {'high':self.high}

    @property
    def hypparams(self):
        return dict(x_m=self.x_m,alpha=self.alpha,low=self.low)

    def log_likelihood(self,x):
        x = np.atleast_1d(x)
        raw = np.where(
            (self.low <= x) & (x < self.high),
            -np.log(self.high - self.low),-np.inf)
        return raw if isinstance(x,np.ndarray) else raw[0]

    def rvs(self,size=[]):
        return np.random.uniform(low=self.low,high=self.high,size=size)

    def resample(self,data=[]):
        self.high = sample_pareto(
            *self._posterior_hypparams(*self._get_statistics(data)))
        return self

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            datamax = data.max()
        else:
            n = sum(d.shape[0] for d in data)
            datamax = \
                max(d.max() for d in data) if n > 0 else -np.inf
        return n, datamax

    def _posterior_hypparams(self,n,datamax):
        return max(datamax,self.x_m), n + self.alpha


class Uniform(UniformOneSided):
    '''
    Models a uniform distribution over [low,high] for parameters low and high.
    The prior is non-conjugate (though it's conditionally conjugate over one
    parameter at a time).

    Likelihood is x ~ U[low,high]
    Prior is -low ~ Pareto(x_m_low,alpha_low)-2*x_m_low
             high ~ Pareto(x_m_high,alpha_high)

    Hyperparameters:
        x_m_low, alpha_low
        x_m_high, alpha_high

    Parameters:
        low, high
    '''
    def __init__(
            self,low=None,high=None,
            x_m_low=None,alpha_low=None,x_m_high=None,alpha_high=None):
        self.low = low
        self.high = high

        self.x_m_low = x_m_low
        self.alpha_low = alpha_low
        self.x_m_high = x_m_high
        self.alpha_high = alpha_high

        have_hypers = not any_none(x_m_low,alpha_low,x_m_high,alpha_high)
        if low is high is None and have_hypers:
            self.resample()  # initialize from prior

    @property
    def params(self):
        return dict(low=self.low,high=self.high)

    @property
    def hypparams(self):
        return dict(
            x_m_low=self.x_m_low,alpha_low=self.alpha_low,
            x_m_high=self.x_m_high,alpha_high=self.alpha_high)

    def resample(self,data=[],niter=5):
        if len(data) == 0:
            self.low = -sample_pareto(-self.x_m_low,self.alpha_low)
            self.high = sample_pareto(self.x_m_high,self.alpha_high)
        else:
            for itr in range(niter):
                # resample high, fixing low
                self.x_m, self.alpha = self.x_m_high, self.alpha_high
                super(Uniform,self).resample(data)
                # tricky: flip data and resample 'high' again
                self.x_m, self.alpha = -self.x_m_low, self.alpha_low
                self.low, self.high = self.high, self.low
                super(Uniform,self).resample(self._flip_data(data))
                self.low, self.high = self.x_m_low - self.high, self.low

    def _flip_data(self,data):
        if isinstance(data,np.ndarray):
            return self.x_m_low - data
        else:
            return list(map(self._flip_data,data))
