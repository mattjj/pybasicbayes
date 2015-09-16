from __future__ import division
from builtins import zip
from builtins import range
__all__ = ['_FixedParamsMixin', 'ProductDistribution']

import numpy as np

from pybasicbayes.abstractions import Distribution, \
    GibbsSampling, MeanField, MeanFieldSVI, MaxLikelihood
from pybasicbayes.util.stats import atleast_2d


class _FixedParamsMixin(Distribution):
    @property
    def num_parameters(self):
        return 0

    def resample(self, *args, **kwargs):
        return self

    def meanfieldupdate(self, *args, **kwargs):
        return self

    def get_vlb(self):
        return 0.

    def copy_sample(self):
        return self


class ProductDistribution(
        GibbsSampling, MeanField, MeanFieldSVI, MaxLikelihood):
    def __init__(self, distns, slices=None):
        self._distns = distns
        self._slices = slices if slices is not None else \
            [slice(i, i+1) for i in range(len(distns))]

    @property
    def params(self):
        return {idx:distn.params for idx, distn in enumerate(self._distns)}

    @property
    def hypparams(self):
        return {idx:distn.hypparams for idx, distn in enumerate(self._distns)}

    @property
    def num_parameters(self):
        return sum(d.num_parameters for d in self._distns)

    def rvs(self,size=[]):
        return np.concatenate(
            [atleast_2d(distn.rvs(size=size))
             for distn in self._distns],axis=-1)

    def log_likelihood(self,x):
        return sum(
            distn.log_likelihood(x[...,sl])
            for distn,sl in zip(self._distns,self._slices))

    ### Gibbs

    def resample(self,data=[]):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.resample(data[...,sl])
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.resample([d[...,sl] for d in data])
        return self

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.max_likelihood(data[...,sl],weights=weights)
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.max_likelihood([d[...,sl] for d in data],weights=weights)
        return self

    ### Mean field

    def get_vlb(self):
        return sum(distn.get_vlb() for distn in self._distns)

    def expected_log_likelihood(self,x):
        return np.sum(
            [distn.expected_log_likelihood(x[...,sl])
             for distn,sl in zip(self._distns,self._slices)], axis=0).ravel()

    def meanfieldupdate(self,data,weights,**kwargs):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfieldupdate(data[...,sl],weights)
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfieldupdate(
                    [d[...,sl] for d in data],weights=weights)
        return self

    def _resample_from_mf(self):
        for distn in self._distns:
            distn._resample_from_mf()

    ### SVI

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfield_sgdstep(
                    data[...,sl],weights,prob,stepsize)
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfield_sgdstep(
                    [d[...,sl] for d in data],weights,prob,stepsize)
        return self
