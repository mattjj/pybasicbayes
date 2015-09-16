from __future__ import print_function
from builtins import range
from builtins import object
import abc
import numpy as np
import copy

import pybasicbayes
from pybasicbayes.util.stats import combinedata
from pybasicbayes.util.text import progprint_xrange
from future.utils import with_metaclass

# NOTE: data is always a (possibly masked) np.ndarray or list of (possibly
# masked) np.ndarrays.

# TODO figure out a data abstraction
# TODO make an exponential family abc to reduce boilerplate

################
#  Base class  #
################

class Distribution(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def rvs(self,size=[]):
        'random variates (samples)'
        pass

    @abc.abstractmethod
    def log_likelihood(self,x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        pass

class BayesianDistribution(with_metaclass(abc.ABCMeta, Distribution)):
    def empirical_bayes(self,data):
        '''
        (optional) set hyperparameters via empirical bayes
        e.g. treat argument as a pseudo-dataset for exponential family
        '''
        raise NotImplementedError

#########################################################
#  Algorithm interfaces for inference in distributions  #
#########################################################

class GibbsSampling(with_metaclass(abc.ABCMeta, BayesianDistribution)):
    @abc.abstractmethod
    def resample(self,data=[]):
        pass

    def copy_sample(self):
        '''
        return an object copy suitable for making lists of posterior samples
        (override this method to prevent copying shared structures into each sample)
        '''
        return copy.deepcopy(self)

    def resample_and_copy(self):
        self.resample()
        return self.copy_sample()

class MeanField(with_metaclass(abc.ABCMeta, BayesianDistribution)):
    @abc.abstractmethod
    def expected_log_likelihood(self,x):
        pass

    @abc.abstractmethod
    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError

class MeanFieldSVI(with_metaclass(abc.ABCMeta, BayesianDistribution)):
    @abc.abstractmethod
    def meanfield_sgdstep(self,expected_suff_stats,prob,stepsize):
        pass

class Collapsed(with_metaclass(abc.ABCMeta, BayesianDistribution)):
    @abc.abstractmethod
    def log_marginal_likelihood(self,data):
        pass

    def log_predictive(self,newdata,olddata):
        return self.log_marginal_likelihood(combinedata((newdata,olddata))) \
                    - self.log_marginal_likelihood(olddata)

    def predictive(self,*args,**kwargs):
        return np.exp(self.log_predictive(*args,**kwargs))

class MaxLikelihood(with_metaclass(abc.ABCMeta, Distribution)):
    @abc.abstractmethod
    def max_likelihood(self,data,weights=None):
        '''
        sets the parameters set to their maximum likelihood values given the
        (weighted) data
        '''
        pass

    @property
    def num_parameters(self):
        raise NotImplementedError

class MAP(with_metaclass(abc.ABCMeta, BayesianDistribution)):
    @abc.abstractmethod
    def MAP(self,data,weights=None):
        '''
        sets the parameters to their MAP values given the (weighted) data
        analogous to max_likelihood but includes hyperparameters
        '''
        pass

class Tempering(BayesianDistribution):
    @abc.abstractmethod
    def log_likelihood(self,data,temperature=1.):
        pass

    @abc.abstractmethod
    def resample(self,data,temperature=1.):
        pass

    def energy(self,data):
        return -self.log_likelihood(data,temperature=1.)

############
#  Models  #
############

# a "model" is differentiated from a "distribution" in this code by latent state
# over data: a model attaches a latent variable (like a label or state sequence)
# to data, and so it 'holds onto' data. Hence the add_data method.

class Model(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def add_data(self,data):
        pass

    @abc.abstractmethod
    def generate(self,keep=True,**kwargs):
        '''
        Like a distribution's rvs, but this also fills in latent state over
        data and keeps references to the data.
        '''
        pass

    def rvs(self,*args,**kwargs):
        return self.generate(*args,keep=False,**kwargs)[0] # 0th component is data, not latent stuff

##################################################
#  Algorithm interfaces for inference in models  #
##################################################

class ModelGibbsSampling(with_metaclass(abc.ABCMeta, Model)):
    @abc.abstractmethod
    def resample_model(self): # TODO niter?
        pass

    def copy_sample(self):
        '''
        return an object copy suitable for making lists of posterior samples
        (override this method to prevent copying shared structures into each sample)
        '''
        return copy.deepcopy(self)

    def resample_and_copy(self):
        self.resample_model()
        return self.copy_sample()

class ModelMeanField(with_metaclass(abc.ABCMeta, Model)):
    @abc.abstractmethod
    def meanfield_coordinate_descent_step(self):
        # returns variational lower bound after update, if available
        pass

    def meanfield_coordinate_descent(self,tol=1e-1,maxiter=250,progprint=False,**kwargs):
        # NOTE: doesn't re-initialize!
        scores = []
        step_iterator = range(maxiter) if not progprint else progprint_xrange(maxiter)
        for itr in step_iterator:
            scores.append(self.meanfield_coordinate_descent_step(**kwargs))
            if scores[-1] is not None and len(scores) > 1:
                if np.abs(scores[-1]-scores[-2]) < tol:
                    return scores
        print('WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter)
        return scores

class ModelMeanFieldSVI(with_metaclass(abc.ABCMeta, Model)):
    @abc.abstractmethod
    def meanfield_sgdstep(self,minibatch,prob,stepsize):
        pass

class _EMBase(with_metaclass(abc.ABCMeta, Model)):
    @abc.abstractmethod
    def log_likelihood(self):
        # returns a log likelihood number on attached data
        pass

    def _EM_fit(self,method,tol=1e-1,maxiter=100,progprint=False):
        # NOTE: doesn't re-initialize!
        likes = []
        step_iterator = range(maxiter) if not progprint else progprint_xrange(maxiter)
        for itr in step_iterator:
            method()
            likes.append(self.log_likelihood())
            if len(likes) > 1:
                if likes[-1]-likes[-2] < tol:
                    return likes
                elif likes[-1] < likes[-2]:
                    # probably oscillation, do one more
                    method()
                    likes.append(self.log_likelihood())
                    return likes
        print('WARNING: EM_fit reached maxiter of %d' % maxiter)
        return likes

class ModelEM(with_metaclass(abc.ABCMeta, _EMBase)):
    def EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.EM_step,tol=tol,maxiter=maxiter)

    @abc.abstractmethod
    def EM_step(self):
        pass

class ModelMAPEM(with_metaclass(abc.ABCMeta, _EMBase)):
    def MAP_EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.MAP_EM_step,tol=tol,maxiter=maxiter)

    @abc.abstractmethod
    def MAP_EM_step(self):
        pass

class ModelParallelTempering(with_metaclass(abc.ABCMeta, Model)):
    @abc.abstractproperty
    def temperature(self):
        pass

    @abc.abstractproperty
    def energy(self):
        pass

    @abc.abstractmethod
    def swap_sample_with(self,other):
        pass

