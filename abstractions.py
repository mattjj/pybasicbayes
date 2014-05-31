import abc
import numpy as np
import copy

from util.stats import combinedata
from util.text import progprint_xrange

# NOTE: data is always a (possibly masked) np.ndarray or list of (possibly
# masked) np.ndarrays.

# TODO figure out a data abstraction
# TODO make an exponential family abc to reduce boilerplate

################
#  Base class  #
################

class Distribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def params(self):
        'distribution parameters'
        pass

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

    def __repr__(self):
        return '%s(params={%s})' % (self.__class__.__name__,self._formatparams(self.params))

    @staticmethod
    def _formatparams(dct):
        return ','.join(('{}:{:3.3G}' if isinstance(val,(int,long,float,complex))
                                        else '{}:{}').format(name,val)
                    for name,val in dct.iteritems()).replace('\n','').replace(',',', ')

class BayesianDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def hypparams(self):
        'hyperparameters define a prior distribution over parameters'
        pass

    def empirical_bayes(self,data):
        '''
        (optional) set hyperparameters via empirical bayes
        e.g. treat argument as a pseudo-dataset for exponential family
        '''
        raise NotImplementedError

    def __repr__(self):
        if not all(v is None for v in self.hypparams.itervalues()):
            return '%s(\nparams={%s},\nhypparams={%s})' % (self.__class__.__name__,
                    self._formatparams(self.params),self._formatparams(self.hypparams))
        else:
            return super(BayesianDistribution,self).__repr__()

#########################################################
#  Algorithm interfaces for inference in distributions  #
#########################################################

class GibbsSampling(BayesianDistribution):
    __metaclass__ = abc.ABCMeta

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

class MeanField(BayesianDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def expected_log_likelihood(self,x):
        pass

    @abc.abstractmethod
    def meanfieldupdate(self,data,weights):
        pass

    def get_vlb(self):
        raise NotImplementedError

class MeanFieldSVI(BayesianDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfield_sgdstep(self,expected_suff_stats,minibatchfrac,stepsize):
        pass

class Collapsed(BayesianDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_marginal_likelihood(self,data):
        pass

    def log_predictive(self,newdata,olddata):
        return self.log_marginal_likelihood(combinedata((newdata,olddata))) \
                    - self.log_marginal_likelihood(olddata)

    def predictive(self,*args,**kwargs):
        return np.exp(self.log_predictive(*args,**kwargs))

class MaxLikelihood(Distribution):
    __metaclass__ = abc.ABCMeta

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

class MAP(BayesianDistribution):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def MAP(self,data,weights=None):
        '''
        sets the parameters to their MAP values given the (weighted) data
        analogous to max_likelihood but includes hyperparameters
        '''
        pass

############
#  Models  #
############

# a "model" is differentiated from a "distribution" in this code by latent state
# over data: a model attaches a latent variable (like a label or state sequence)
# to data, and so it 'holds onto' data. Hence the add_data method.

class Model(object):
    __metaclass__ = abc.ABCMeta

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

class ModelGibbsSampling(Model):
    __metaclass__ = abc.ABCMeta

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

class ModelMeanField(Model):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfield_coordinate_descent_step(self):
        # returns variational lower bound after update, if available
        pass

    def meanfield_coordinate_descent(self,tol=1e-1,maxiter=250,progprint=False,**kwargs):
        # NOTE: doesn't re-initialize!
        scores = []
        step_iterator = xrange(maxiter) if not progprint else progprint_xrange(maxiter)
        for itr in step_iterator:
            scores.append(self.meanfield_coordinate_descent_step(**kwargs))
            if scores[-1] is not None and len(scores) > 1:
                if np.abs(scores[-1]-scores[-2]) < tol:
                    return scores
        print 'WARNING: meanfield_coordinate_descent hit maxiter of %d' % maxiter
        return scores

class ModelMeanFieldSVI(Model):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize):
        pass

class _EMBase(Model):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_likelihood(self):
        # returns a log likelihood number on attached data
        pass

    def _EM_fit(self,method,tol=1e-1,maxiter=100,progprint=False):
        # NOTE: doesn't re-initialize!
        likes = []
        step_iterator = xrange(maxiter) if not progprint else progprint_xrange(maxiter)
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
        print 'WARNING: EM_fit reached maxiter of %d' % maxiter
        return likes

class ModelEM(_EMBase):
    __metaclass__ = abc.ABCMeta

    def EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.EM_step,tol=tol,maxiter=maxiter)

    @abc.abstractmethod
    def EM_step(self):
        pass

class ModelMAPEM(_EMBase):
    __metaclass__ = abc.ABCMeta

    def MAP_EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.MAP_EM_step,tol=tol,maxiter=maxiter)

    @abc.abstractmethod
    def MAP_EM_step(self):
        pass

