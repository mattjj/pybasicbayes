import abc
import numpy as np
from warnings import warn

from util.stats import combinedata

# NOTE: data is always a (possibly masked) np.ndarray or list of (possibly
# masked) np.ndarrays.

########################
#  Distribution types  #
########################

class Distribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def rvs(self,size=[]):
        '''
        random variates (samples)
        '''
        pass

    @abc.abstractmethod
    def log_likelihood(self,x):
        '''
        log likelihood (either log probability mass function or log probability
        density function)
        '''
        pass

class DurationDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_sf(self,x):
        '''
        log survival function, defined by log_sf(x) = log(1-cdf(x)) where
        cdf(x) = P[X \leq x]
        '''
        pass

#########################################################
#  Algorithm interfaces for inference in distributions  #
#########################################################

class GibbsSampling(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def resample(self,data=[]):
        pass

class MeanField(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def expected_log_likelihood(self,x):
        pass

    @abc.abstractmethod
    def meanfieldupdate(self,data,weights):
        pass

class Collapsed(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def marginal_log_likelihood(self,data):
        pass

    def predictive(self,newdata,olddata):
        return np.exp(self.marginal_log_likelihood(combinedata((newdata,olddata)))
                    - self.marginal_log_likelihood(olddata))

class MaxLikelihood(Distribution):
    __metaclass__ = abc.ABCMeta

    # should be an abstract classmethod, but Python 2.7.3 doesn't have them!
    @abc.abstractmethod
    def max_likelihood_fit(cls,data):
        # returns an instance of the object with hyperparameter members set to
        # None
        pass

############
#  Models  #
############

# what differentiates a "model" from a "distribution" in this code is latent
# state over data: a model attaches a latent variable (like a label or state
# sequence) to data, and so it 'holds onto' data. Hence the add_data method.

class Model(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_data(self,data):
        pass

    @abc.abstractmethod
    def generate(self,keep=True,**kwargs):
        '''
        Like a distribution's rvs, but this also fills in latent state over
        data, and keeps references to the data.
        '''
        pass

    def rvs(self,*args,**kwargs):
        return self.generate(*args,keep=False,**kwargs)

##################################################
#  Algorithm interfaces for inference in models  #
##################################################

class ModelGibbsSampling(Model):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def resample_model(self): # TODO niter?
        pass

class ModelMeanField(Model):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfield_coordinate_descent_step(self): # TODO convergence tol?
        pass

    # TODO VLB?

# TODO parallel gibbs sampling model algorithm interface... uses ipython and
# (hopefully custom) pickling
