from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.stats as stats
import scipy.special as special
from matplotlib import pyplot as plt
import abc

from pyhsmm.abstractions import Distribution, GibbsSampling, MeanField, Collapsed
from pyhsmm.util.stats import sample_niw, sample_discrete, \
        sample_discrete_from_log, getdatasize, combinedata

class Gaussian(GibbsSampling, MeanField, Collapsed, Distribution):
    '''
    Multivariate Gaussian observation distribution class. NOTE: Only works for 2
    or more dimensions. For a scalar Gaussian, use one of the scalar classes.
    Uses a conjugate Normal/Inverse-Wishart prior.

    Hyperparameters mostly follow Gelman et al.'s notation in Bayesian Data
    Analysis, except sigma_0 is proportional to expected covariance matrix:
        nu_0, sigma_0
        mu_0, kappa_0

    Parameters are mean and covariance matrix:
        mu, sigma
    '''

    def __init__(self,mu_0,sigma_0,kappa_0,nu_0,mu=None,sigma=None):
        self._mu_mf    = self.mu_0    = mu_0
        self._sigma_mf = self.sigma_0 = sigma_0
        self._kappa_mf = self.kappa_0 = kappa_0
        self._nu_mf    = self.nu_0    = nu_0

        self.D = mu_0.shape[0]
        assert sigma_0.shape == (self.D,self.D) and self.D >= 2

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma

    def rvs(self,size=[]):
        return np.random.multivariate_normal(mean=self.mu,cov=self.sigma,size=size)

    def log_likelihood(self,x):
        mu, sigma, D = self.mu, self.sigma, self.D
        x = np.reshape(x,(-1,D)) - mu
        return -1./2. * (x * np.linalg.solve(sigma,x.T).T).sum(1) \
                - D/2*np.log((2*np.pi) * np.sqrt(np.linalg.det(sigma)))

    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, sigma_0, kappa_0, nu_0 = self.mu_0, self.sigma_0, self.kappa_0, self.nu_0
        if n > 0:
            mu_n = self.kappa_0 / (self.kappa_0 + n) * self.mu_0 + n / (self.kappa_0 + n) * xbar
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            sigma_n = self.sigma_0 + sumsq + \
                    self.kappa_0*n/(self.kappa_0+n) * np.outer(xbar-self.mu_0,xbar-self.mu_0)

            return mu_n, sigma_n, kappa_n, nu_n
        else:
            return mu_0, sigma_0, kappa_0, nu_0

    ### Gibbs sampling

    def resample(self,data=[]):
        self.mu, self.sigma = sample_niw(*self._posterior_hypparams(*self._get_statistics(data)))

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        D = self.D
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = np.reshape(data,(-1,D)).mean(0)
                centered = data - xbar
                sumsq = np.dot(centered.T,centered)
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,(np.reshape(d,(-1,D))-xbar))
                        for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq

    ### Mean Field

    def meanfieldupdate(self,data,weights):
        assert getdatasize(data) > 0
        self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights))

    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf
        D = self.D

        x = np.reshape(x,(-1,D)) - mu_n

        # see Eq. 10.65 in Bishop
        loglmbdatilde = special.digamma((nu_n+np.arange(D))/2).sum() \
                + D*np.log(2) - np.log(np.linalg.det(sigma_n))

        # see Eq. 10.67 in Bishop
        return np.sqrt(loglmbdatilde) - D/(2*kappa_n) - nu_n/2 * \
                (np.linalg.solve(sigma_n,x.T).T * x).sum(1)

    def _get_weighted_statistics(self,data,weights):
        # NOTE: _get_statistics is special case with all weights being 1
        # this is kept as a separate method for speed and modularity
        D = self.D

        assert (isinstance(data,np.ndarray) and isinstance(weights,np.ndarray)
                and weights.ndim == 1 and np.reshape(data,(-1,D)).shape[0] == weights.shape[0]) \
                        or \
                        (isinstance(data,list) and isinstance(weights,list) and
                                all(isinstance(d,np.ndarray) and isinstance(w,np.ndarray)
                                    and w.ndim == 1 and np.reshape(d,(-1,D)).shape[0] == w.shape[0])
                                for w,d in zip(weights,data))

        if isinstance(data,np.ndarray):
            neff = weights.sum()
            xbar = np.dot(weights,np.reshape(data,(-1,D))) / neff
            centered = np.reshape(data,(-1,D)) - xbar
            sumsq = np.dot(centered.T,(weights[:,na] * centered))
        else:
            neff = sum(w.sum() for w in weights)
            xbar = sum(np.dot(w,np.reshape(d,(-1,D))) for w,d in zip(weights,data)) / neff
            sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,w[:,na]*(np.reshape(d,(-1,D))-xbar))
                    for w,d in zip(weights,data))
        return neff, xbar, sumsq

    ### Collapsed

    def marginal_log_likelihood(self,data):
        n, D = getdatasize(data), self.D
        return self.prior_log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self.prior_log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
                - n*D/2 * np.log(2*np.pi)

    def prior_log_partition_function(self,mu,sigma,kappa,nu):
        D = self.D
        return nu*D/2*np.log(2) + special.multigammaln(nu/2,D) + D/2*np.log(2*np.pi/kappa) \
                - nu/2*np.log(np.linalg.det(sigma))

    ### Misc

    @classmethod
    def _plot_setup(cls,instance_list):
        # must set cls.vecs to be a reasonable 2D space to project onto
        # so that the projection is consistent across instances
        # for now, i'll just make it random if there are more than 2 dimensions
        assert len(instance_list) > 0
        assert len(set([len(o.mu) for o in instance_list])) == 1, \
                'must have consistent dimensions across instances'
        dim = len(instance_list[0].mu)
        if dim > 2:
            vecs = np.random.randn((dim,2))
            vecs /= np.sqrt((vecs**2).sum())
        else:
            vecs = np.eye(2)

        for o in instance_list:
            o.global_vecs = vecs

    def plot(self,data=None,color='b'):
        from pyhsmm.util.plot import project_data, plot_gaussian_projection, pca
        # if global projection vecs exist, use those
        # otherwise, when dim>2, do a pca on the data
        try:
            vecs = self.global_vecs
        except AttributeError:
            dim = len(self.mu)
            if dim == 2:
                vecs = np.eye(2)
            else:
                assert dim > 2
                vecs = pca(data,num_components=2)

        if data is not None:
            projected_data = project_data(data,vecs)
            plt.plot(projected_data[:,0],projected_data[:,1],marker='.',linestyle=' ',color=color)

        plot_gaussian_projection(self.mu,self.sigma,vecs,color=color)


class DiagonalGaussian(GibbsSampling, Distribution):
    '''
    Product of normal-inverse-gamma priors over mu (mean vector) and sigmas
    (vector of scalar variances).

    The prior follows
        sigmas     ~ InvGamma(alphas_0,betas_0) iid
        mu | sigma ~ N(mu_0,1/nus_0 * diag(sigmas))

    It allows placing different prior hyperparameters on different components.
    '''
    # doesn't inherit from Gaussian because it allows different prior
    # hyperparameters on different components

    def __init__(self,mu_0,nus_0,alphas_0,betas_0,mu=None,sigmas=None):
        D = mu_0.shape[0]
        self.mu_0 = mu_0
        # all the s's refer to the fact that these are vectors of length
        # len(mu_0) OR scalars
        self.nus_0 = nus_0
        self.alphas_0 = alphas_0
        self.betas_0 = betas_0

        self.D = D

        if mu is None or sigmas is None:
            self.resample()
        else:
            self.mu = mu
            self.sigmas = sigmas

    def rvs(self,size=[]):
        size = np.array(size,ndmin=1)
        return np.sqrt(self.sigmas)*np.random.normal(size=np.concatenate((size,self.mu.shape))) + self.mu

    def log_likelihood(self,x):
        mu, sigmas = self.mu, self.sigmas
        x = np.reshape(x,(-1,self.D))
        return (-0.5*((x-mu)**2/sigmas) - np.log(np.sqrt(2*np.pi*sigmas))).sum(1)

    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, nus_0, alphas_0, betas_0 = self.mu_0, self.nus_0, self.alphas_0, self.betas_0
        if n > 0:
            nus_n = n + nus_0
            alphas_n = alphas_0 + n/2
            betas_n = betas_0 + 1/2*sumsq + n*nus_0/(n+nus_0) * 1/2*(xbar - mu_0)**2
            mu_n = (n*xbar + nus_0*mu_0)/(n+nus_0)

            return mu_n, nus_n, alphas_n, betas_n
        else:
            return mu_0, nus_0, alphas_0, betas_0

    ### Gibbs sampling

    def resample(self,data=[]):
        mu_n, nus_n, alphas_n, betas_n = self._posterior_hypparams(*self._get_statistics(data))
        self.sigmas = 1/np.random.gamma(alphas_n,scale=1/betas_n)
        self.mu = np.sqrt(self.sigmas/nus_n)*np.random.randn(self.D) + mu_n

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        D = self.D
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                data = np.reshape(data,(-1,D))
                xbar = data.mean(0)
                centered = data - xbar
                sumsq = np.dot(centered.T,centered)
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(((np.reshape(d,(-1,D)) - xbar)**2).sum(0) for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq


class IsotropicGaussian(GibbsSampling, Distribution):
    '''
    Normal-Inverse-Gamma prior over mu (mean vector) and sigma (scalar
    variance). Essentially, all coordinates of all observations inform the
    variance.

    The prior follows
        sigma      ~ InvGamma(alpha_0,beta_0)
        mu | sigma ~ N(mu_0,sigma/nu_0 * I)
    '''

    def __init__(self,mu_0,nu_0,alpha_0,beta_0,mu=None,sigma=None):
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.D = mu_0.shape[0]

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma

    def rvs(self,size=[]):
        return np.sqrt(self.sigma)*np.random.normal(size=tuple(size)+self.mu.shape) + self.mu

    def log_likelihood(self,x):
        mu, sigma, D = self.mu, self.sigma, self.D
        x = np.reshape(x,(-1,D))
        return (-0.5*((x-mu)**2).sum(1)/sigma - D*np.log(np.sqrt(2*np.pi*sigma)))

    def _posterior_hypparams(self,n,xbar,sumsq):
        D = self.D
        mu_0, nu_0, alpha_0, beta_0 = self.mu_0, self.nu_0, self.alpha_0, self.beta_0
        if n > 0:
            nu_n = D*n + nu_0
            alpha_n = alpha_0 + D*n/2
            beta_n = beta_0 + 1/2*sumsq + (n*D*nu_0)/(n*D+nu_0) * 1/2 * ((xbar - mu_0)**2).sum()
            mu_n = (n*xbar + nu_0*mu_0)/(n+nu_0)

            return mu_n, nu_n, alpha_n, beta_n
        else:
            return mu_0, nu_0, alpha_0, beta_0

    ### Gibbs sampling

    def resample(self,data=[]):
        mu_n, nu_n, alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        self.sigma = 1/np.random.gamma(alpha_n,scale=1/beta_n)
        self.mu = np.sqrt(self.sigma/nu_n)*np.random.randn(self.D)+mu_n

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        D = self.D
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                data = np.reshape(data,(-1,D))
                xbar = data.mean(0)
                sumsq = ((data-xbar)**2).sum()
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(((np.reshape(data,(-1,D)) - xbar)**2).sum() for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq


class ScalarGaussian(Distribution):
    '''
    Abstract class for all scalar Gaussians.
    '''
    __metaclass__ = abc.ABCMeta

    def rvs(self,size=None):
        return np.sqrt(self.sigmasq)*np.random.normal(size=size)+self.mu

    def log_likelihood(self,x):
        assert x.ndim == 2
        assert x.shape[1] == 1
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).flatten()

    def __repr__(self):
        return 'ScalarGaussian(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

    def plot(self,*args,**kwargs): # TODO
        raise NotImplementedError


class ScalarGaussianNIX(ScalarGaussian, GibbsSampling, Collapsed):
    '''
    Conjugate Normal-Inverse-ChiSquared prior. (Another parameterization is the
    Normal-Inverse-Gamma; that's not implemented, but the hyperparameters can be
    mapped to NIX form.)
    '''
    def __init__(self,mu_0,kappa_0,sigmasq_0,nu_0,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.sigmasq_0 = sigmasq_0
        self.nu_0 = nu_0

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        self.resample()

    def _posterior_hypparams(self,n,ybar,sumsqc):
        mu_0, kappa_0, sigmasq_0, nu_0 = self.mu_0, self.kappa_0, self.sigmasq_0, self.nu_0
        if n > 0:
            kappa_n = kappa_0 + n
            mu_n = (kappa_0 * mu_0 + n * ybar) / kappa_n
            nu_n = nu_0 + n
            sigmasq_n = 1/nu_n * (nu_0 * sigmasq_0 + sumsqc + kappa_0 * n / (kappa_0 + n) * (ybar - mu_0)**2)

            return mu_n, kappa_n, sigmasq_n, nu_n
        else:
            return mu_0, kappa_0, sigmasq_0, nu_0

    ### Gibbs sampling

    def resample(self,data=[]):
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))

        self.sigmasq = nu_n * sigmasq_n / stats.chi2.rvs(nu_n)
        self.mu = np.sqrt(self.sigmasq / kappa_n) * np.random.randn() + mu_n

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq

    def _get_statistics(cls,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all((isinstance(d,np.ndarray))
                    for d in data))

        if isinstance(data,np.ndarray):
            n = data.size
            ybar = data.mean()
            sumsqc = ((data-ybar)**2).sum()
        else:
            n = sum(d.size for d in data)
            ybar = sum(d.sum() for d in data)/n
            sumsqc = sum(np.sum((d-ybar)**2) for d in data)

        return n, ybar, sumsqc

    ### Collapsed

    def marginal_log_likelihood(self,data):
        n = getdatasize(data)
        mu_0, kappa_0, sigmasq_0, nu_0 = self.mu_0, self.kappa_0, self.sigmasq_0, self.nu_0
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        return np.exp(special.gammaln(nu_n/2) - special.gammaln(nu_0/2) \
                + 0.5*(np.log(kappa_0) - np.log(kappa_n) \
                       + nu_0 * (np.log(nu_0) + np.log(sigmasq_0)) \
                         - nu_n * (np.log(nu_n) + np.log(sigmasq_n)) \
                       - n*np.log(np.pi)))

    def predictive_single(self,y,olddata):
        # mostly for testing or speed
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata))
        return stats.t.pdf(y,nu_n,loc=mu_n,scale=np.sqrt((1+kappa_n)*sigmasq_n/kappa_n))


class ScalarGaussianFixedvar(ScalarGaussian, GibbsSampling):
    '''
    Conjugate normal prior on mean.
    '''
    def __init__(self,mu_0,tausq_0,sigmasq,mu=None,mubin=None,sigmasqbin=None):
        self.mu_0 = mu_0
        self.tausq_0 = tausq_0
        self.sigmasq = sigmasq
        self.mubin = mubin

        # set only once
        if sigmasqbin is not None:
            sigmasqbin[...] = sigmasq
        if mu is None:
            self.resample()
        else:
            self.mu = mu
            if mubin is not None:
                mubin[...] = mu

    def _posterior_hypparams(self,n,xbar):
        mu_0, tausq_0 = self.mu_0, self.tausq_0
        sigmasq = self.sigmasq
        if n > 0:
            tausq_n = 1/(1/tausq_0 + n/sigmasq)
            mu_n = (mu_0/tausq_0 + n*xbar/sigmasq)*tausq_n

            return mu_n, tausq_n
        else:
            return mu_0, tausq_0

    def resample(self,data=[]):
        mu_n, tausq_n = self._posterior_hypparams(*self._get_statistics(data))
        self.mu = np.sqrt(tausq_n)*np.random.randn()+mu_n

        if self.mubin is not None:
            self.mubin[...] = self.mu

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = data.mean()
            else:
                xbar = sum(d.sum() for d in data)/n
        else:
            xbar = None
        return n, xbar


class Multinomial(GibbsSampling, Distribution): # TODO meanfield
    '''
    This class represents a multinomial distribution over labels, where the
    parameter is weights and the prior is a Dirichlet distribution.
    For example, if len(alphav_0) == 3, then five samples may look like
    [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls.

    This can be used as a weak limit approximation for a DP, particularly by
    calling __init__ with alpha_0 and K arguments, in which case the prior will be
    a symmetric Dirichlet with K components and parameter alpha_0/K; K is then the
    weak limit approximation parameter.

    Hyperparaemters:
        alphav_0 (vector) OR alpha_0 (scalar) and K

    Parameters:
        weights, a vector encoding a discrete pmf
    '''
    def __init__(self,alphav_0=None,weights=None,alpha_0=None,K=None):
        assert (isinstance(alphav_0,np.ndarray) and alphav_0.ndim == 1) ^ \
                (K is not None and alpha_0 is not None)

        if isinstance(alphav_0,np.ndarray):
            self.alphav_0 = alphav_0
            self.K = alphav_0.shape[0]
        else:
            self.K = K
            self.alphav_0 = np.ones(K)*alpha_0/K

        if weights is not None:
            self.weights = weights
        else:
            self.resample()

    def rvs(self,size=[]):
        return sample_discrete(self.weights,size)

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    def _posterior_hypparams(self,counts):
        return self.alphav + counts,

    ### Gibbs sampling

    def resample(self,data=[]):
        self.weights = np.random.dirichlet(*self._posterior_hypparams(*self.get_statistics(data)))

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        K = self.K
        if isinstance(data,np.ndarray):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)
        return counts,




# TODO TODO below here

# TODO make a collapsed crp mixture and a directassignment crp mixture
# class DiagonalGaussianNonconj(DiagonalGaussian): # TODO for jackie subhmms
#     '''
#     sigmasq_0, alpha_0, beta_0 parameters can either be vectors of same length
#     as mu_0 (to have different parameters along each dimension) or scalars
#     '''

#     # TODO make argument names consistent with diagonal_gaussian
#     def __init__(self,mu_0,sigmasq_0,alpha_0,beta_0,mu=None,sigmas=None):
#         self.mu_0 = mu_0
#         self.sigmasq_0 = sigmasq_0 if type(sigmasq_0) == np.ndarray else sigmasq_0 * np.ones(len(mu_0))
#         self.alpha_0 = alpha_0 if type(alpha_0) == np.ndarray else alpha_0 * np.ones(len(mu_0))
#         self.beta_0 = beta_0 if type(beta_0) == np.ndarray else beta_0 * np.ones(len(mu_0))

#         if mu is None or sigmas is None:
#             self.resample()
#         else:
#             self.mu = mu
#             self.sigmas = sigmas

#     def resample(self,data=np.array([]),niter=5,**kwargs):
#         n = float(len(data))
#         k = len(self.mu_0)
#         if n == 0:
#             self.mu = np.sqrt(self.sigmasq_0)*np.random.normal(self.mu_0.shape) + self.mu_0
#             self.sigmas = stats.invgamma.rvs(self.alpha_0,scale=self.beta_0,size=len(self.mu_0)) # size needed, apparent scipy bug
#         else:
#             for itr in range(niter):
#                 # resample mean given data and var
#                 mu_n = (self.mu_0/self.sigmasq_0 + data.sum(0)/self.sigmas)/(1/self.sigmasq_0 + n/self.sigmas)
#                 sigmasq_n = 1/(1/self.sigmasq_0 + n/self.sigmas)
#                 self.mu = np.sqrt(sigmasq_n)*np.random.normal(self.mu_0.shape)+mu_n
#                 # resample var given data and mean
#                 alpha_n = self.alpha_0 + n/2
#                 beta_n = self.beta_0 + ((data-self.mu)**2).sum(0)/2
#                 self.sigmas = stats.invgamma.rvs(alpha_n,scale=beta_n,size=len(self.mu_0)) # size needed, apparent scipy bug



# class IndicatorMultinomial(Multinomial):
#     '''
#     This class represents a multinomial distribution in an indicator/count form.
#     For example, if len(alpha_vec) == 3, then five samples worth of indicator
#     data may look like
#     [[0,1,0],
#      [1,0,0],
#      [1,0,0],
#      [0,0,1],
#      [0,1,0]]
#     Each row is an indicator of a sample, and summing over rows gives counts.

#     Based on the way the methods are written, the data rows may also be count
#     arrays themselves. The same sample set as in the previous example can also
#     be represented as

#     [[2,2,1]]

#     or

#     [[1,1,1],
#      [1,1,0]]

#     etc.

#     Hyperparameters: alpha_vec
#     Parameters: discrete, which is a vector encoding of a discrete
#     probability distribution
#     '''

#     # TODO collapsed version

#     def resample(self,data=np.array([]),**kwargs):
#         if data.size == 0:
#             counts = np.zeros(self.alpha_vec.shape)
#         elif data.ndim == 2:
#             counts = data.sum(0)
#         else:
#             counts = data
#         self._resample_given_counts(counts)

#     def log_likelihood(self,x):
#         assert x.ndim == 2
#         assert x.shape[1] == len(self.discrete)
#         return (x * np.log(self.discrete)).sum(1)

#     def rvs(self,size=0):
#         assert type(size) == type(0)
#         label_data = multinomial.rvs(self,size=size)
#         out = np.zeros((size,len(self.alpha_vec)))
#         out[np.arange(out.shape[0]),label_data] = 1
#         return out

#     @classmethod
#     def test(cls):
#         # I've tested this by hand
#         raise NotImplementedError

# class ScalarGaussianNonconj(ScalarGaussian, GibbsSampling):
#     def __init__(self,mu_0,sigmasq_0,alpha,beta,mu=None,sigmasq=None,mubin=None,sigmasqbin=None):
#         self.mu_0 = mu_0
#         self.sigmasq_0 = sigmasq_0
#         self.alpha = alpha
#         self.beta = beta

#         self.mubin = mubin
#         self.sigmasqbin = sigmasqbin

#         if mu is None or sigmasq is None:
#             self.resample()
#         else:
#             self.mu = mu
#             self.sigmasq = sigmasq

#     def resample(self,data=np.array([[]]),niter=10):
#         if data.size == 0:
#             # sample from prior
#             self.mu = np.sqrt(self.sigmasq_0)*np.random.randn()+self.mu_0
#             self.sigmasq = stats.invgamma.rvs(self.alpha,scale=self.beta)
#         else:
#             assert data.ndim == 2
#             assert data.shape[1] == 1
#             n = len(data)
#             for iter in xrange(niter):
#                 # resample mean given data and var
#                 mu_n = (self.mu_0/self.sigmasq_0 + data.sum()/self.sigmasq)/(1/self.sigmasq_0 + n/self.sigmasq)
#                 sigmasq_n = 1/(1/self.sigmasq_0 + n/self.sigmasq)
#                 self.mu = np.sqrt(sigmasq_n)*np.random.randn()+mu_n
#                 #resample variance given data and mean
#                 alpha_n = self.alpha+n/2
#                 beta_n = self.beta+((data-self.mu)**2).sum()/2
#                 self.sigmasq = stats.invgamma.rvs(alpha_n,scale=beta_n)

#         if self.mubin is not None and self.sigmasqbin is not None:
#             self.mubin[...] = self.mu
#             self.sigmasqbin[...] = self.sigmasq

#     def __repr__(self):
#         return 'gaussian_scalar_nonconj(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

# class ScalarGaussianNonconjGelparams(ScalarGaussian, GibbsSampling): # TODO
# next
#     # TODO factor out some stuff into scalar gaussian base
#     # uses parameters from Gelman's Bayesian Data Analysis
#     def __init__(self,mu_0,tausq_0,sigmasq_0,nu_0,mu=None,sigmasq=None,mubin=None,sigmasqbin=None):
#         self.mu_0 = mu_0
#         self.tausq_0 = tausq_0
#         self.sigmasq_0 = sigmasq_0
#         self.nu_0 = nu_0

#         self.mubin = mubin
#         self.sigmasqbin = sigmasqbin

#         if mu is None or sigmasq is None:
#             self.resample()
#         else:
#             self.mu = mu
#             self.sigmasq = sigmasq
#             if mubin is not None and sigmasqbin is not None:
#                 mubin[...] = mu
#                 sigmasqbin[...] = sigmasq

#     def resample(self,data=np.array([[]]),niter=10):
#         if data.size == 0:
#             # sample from prior
#             self.mu = np.sqrt(self.tausq_0)*np.random.randn()+self.mu_0
#             self.sigmasq = self.nu_0 * self.sigmasq_0 / stats.chi2.rvs(self.nu_0)
#         else:
#             assert data.ndim == 2 or data.ndim == 1 # TODO why is this important?
#             data = np.reshape(data,(-1,1))
#             n = len(data)
#             mu_hat = data.mean()
#             for iter in xrange(niter):
#                 # resample mean given data and var
#                 mu_n = (self.mu_0/self.tausq_0 + n*mu_hat/self.sigmasq)/(1/self.tausq_0 + n/self.sigmasq)
#                 tausq_n = 1/(1/self.tausq_0 + n/self.sigmasq)
#                 self.mu = np.sqrt(tausq_n)*np.random.randn()+mu_n
#                 #resample variance given data and mean
#                 v = np.var(data - self.mu)
#                 nu_n = self.nu_0 + n
#                 sigmasq_n = (self.nu_0 * self.sigmasq_0 + n*v)/(self.nu_0 + n)
#                 self.sigmasq = nu_n * sigmasq_n / stats.chi2.rvs(nu_n)

#         if self.mubin is not None and self.sigmasqbin is not None:
#             self.mubin[...] = self.mu
#             self.sigmasqbin[...] = self.sigmasq

#     def __repr__(self):
#         return 'gaussian_scalar_nonconj(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

