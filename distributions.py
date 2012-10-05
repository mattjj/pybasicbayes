from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.stats as stats
import scipy.special as special
from matplotlib import pyplot as plt
import abc
from warnings import warn

from abstractions import Distribution, GibbsSampling,\
        MeanField, Collapsed
from util.stats import sample_niw, invwishart_entropy,\
        invwishart_log_partitionfunction, sample_discrete,\
        sample_discrete_from_log, getdatasize, flattendata

################
#  Continuous  #
################

class Gaussian(GibbsSampling, MeanField, Collapsed):
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
        self.mu_0    = mu_0
        self.sigma_0 = sigma_0
        self.kappa_0 = kappa_0
        self.nu_0    = nu_0

        self.D = mu_0.shape[0]
        assert sigma_0.shape == (self.D,self.D) and self.D >= 2

        if mu is None or sigma is None:
            self.resample()
        else:
            self.mu = mu
            self.sigma = sigma
        self._mu_mf = self.mu
        self._sigma_mf = self.sigma
        self._kappa_mf = kappa_0
        self._nu_mf = nu_0

    def rvs(self,size=None):
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
        self._mu_mf, self._sigma_mf = self.mu, self.sigma = \
                sample_niw(*self._posterior_hypparams(*self._get_statistics(data)))

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

    # NOTE my sumsq is Nk*Sk

    def meanfieldupdate(self,data,weights):
        assert getdatasize(data) > 0
        # update
        self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights))
        self.mu, self.sigma = self._mu_mf, self._sigma_mf/(self._nu_mf - self.D - 1) # for plotting

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        D = self.D
        loglmbdatilde = self._loglmbdatilde()
        # see Eq. 10.77 in Bishop
        q_entropy = -1 * (0.5 * (loglmbdatilde + self.D * (np.log(self._kappa_mf/(2*np.pi))-1)) \
                - invwishart_entropy(self._sigma_mf,self._nu_mf))
        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde \
                - D*self.kappa_0/self._kappa_mf - self.kappa_0*self._nu_mf*\
                np.dot(self._mu_mf - self.mu_0,np.linalg.solve(self._sigma_mf,self._mu_mf - self.mu_0))) \
                - invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
                + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self._nu_mf*\
                np.linalg.solve(self._sigma_mf,self.sigma_0).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf
        D = self.D
        x = np.reshape(x,(-1,D)) - mu_n # x is now centered

        # see Eq. 10.67 in Bishop
        return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                (np.linalg.solve(sigma_n,x.T).T * x).sum(1)

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        return special.digamma((self._nu_mf-np.arange(self.D))/2).sum() \
                + self.D*np.log(2) - np.linalg.slogdet(self._sigma_mf)[1]

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
            if neff > 0:
                xbar = np.dot(weights,np.reshape(data,(-1,D))) / neff
                centered = np.reshape(data,(-1,D)) - xbar
                sumsq = np.dot(centered.T,(weights[:,na] * centered))
            else:
                xbar, sumsq = None, None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                xbar = sum(np.dot(w,np.reshape(d,(-1,D))) for w,d in zip(weights,data)) / neff
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,w[:,na]*(np.reshape(d,(-1,D))-xbar))
                        for w,d in zip(weights,data))
            else:
                xbar, sumsq = None, None
        return neff, xbar, sumsq

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n, D = getdatasize(data), self.D
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
                - n*D/2 * np.log(2*np.pi)

    def _log_partition_function(self,mu,sigma,kappa,nu):
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

    def plot(self,data=None,color='b',plot_params=True):
        from util.plot import project_data, plot_gaussian_projection, pca
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

        if plot_params:
            plot_gaussian_projection(self.mu,self.sigma,vecs,color=color)


class DiagonalGaussian(GibbsSampling):
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

    def rvs(self,size=None):
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


class IsotropicGaussian(GibbsSampling):
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

    def rvs(self,size=None):
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
        x = np.reshape(x,(-1,1))
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).flatten()

    def __repr__(self):
        return self.__class__.__name__ + '(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

    @classmethod
    def _plot_setup(cls,instance_list):
        pass

    def plot(self,data=None,color='b',plot_params=True):
        if data is not None:
            pass
        raise NotImplementedError


class ScalarGaussianNIX(ScalarGaussian, GibbsSampling, Collapsed):
    '''
    Conjugate Normal-(Scaled-)Inverse-ChiSquared prior. (Another parameterization is the
    Normal-Inverse-Gamma.)
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

        self.sigmasq = nu_n * sigmasq_n / np.random.chisquare(nu_n)
        self.mu = np.sqrt(self.sigmasq / kappa_n) * np.random.randn() + mu_n

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all((isinstance(d,np.ndarray))
                    for d in data)) or \
                (isinstance(data,int) or isinstance(data,float))

        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                ybar = data.mean()
                sumsqc = ((data-ybar)**2).sum()
            elif isinstance(data,list):
                ybar = sum(d.sum() for d in data)/n
                sumsqc = sum(np.sum((d-ybar)**2) for d in data)
            else:
                ybar = data
                sumsqc = 0
        else:
            ybar = None
            sumsqc = None

        return n, ybar, sumsqc

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n = getdatasize(data)
        mu_0, kappa_0, sigmasq_0, nu_0 = self.mu_0, self.kappa_0, self.sigmasq_0, self.nu_0
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        return special.gammaln(nu_n/2) - special.gammaln(nu_0/2) \
                + 0.5*(np.log(kappa_0) - np.log(kappa_n) \
                       + nu_0 * (np.log(nu_0) + np.log(sigmasq_0)) \
                         - nu_n * (np.log(nu_n) + np.log(sigmasq_n)) \
                       - n*np.log(np.pi))

    def log_predictive_single(self,y,olddata):
        # mostly for testing or speed
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata))
        return stats.t.logpdf(y,nu_n,loc=mu_n,scale=np.sqrt((1+kappa_n)*sigmasq_n/kappa_n))


class ScalarGaussianNonconjNIX(ScalarGaussian, GibbsSampling):
    '''
    Non-conjugate separate priors on mean and variance parameters, via
    mu ~ Normal(mu_0,tausq_0)
    sigmasq ~ (Scaled-)Inverse-ChiSquared(sigmasq_0,nu_0)
    '''
    def __init__(self,mu_0,tausq_0,sigmasq_0,nu_0,mubin=None,sigmasqbin=None):
        self.mu_0, self.tausq_0 = mu_0, tausq_0
        self.sigmasq_0, self.nu_0 = sigmasq_0, nu_0

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        self.resample()

    def resample(self,data=[],niter=30):
        n = getdatasize(data)
        if n > 0:
            data = flattendata(data)
            datasum = data.sum()
            nu_n = self.nu_0 + n
            for itr in range(niter):
                # resample mean
                tausq_n = 1/(1/self.tausq_0 + n/self.sigmasq)
                mu_n = tausq_n*(self.mu_0/self.tausq_0 + datasum/self.sigmasq)
                self.mu = np.sqrt(tausq_n)*np.random.normal() + mu_n
                # resample variance
                sigmasq_n = (self.nu_0*self.sigmasq_0 + ((data-self.mu)**2).sum())/(nu_n)
                self.sigmasq = sigmasq_n*nu_n/np.random.chisquare(nu_n)
        else:
            self.mu = np.sqrt(self.tausq_0) * np.random.normal() + self.mu_0
            self.sigmasq = self.sigmasq_0*self.nu_0/np.random.chisquare(self.nu_0)

        if self.mubin is not None and self.sigmasqbin is not None:
            self.mubin[...] = self.mu
            self.sigmasqbin[...] = self.sigmasq


class ScalarGaussianFixedvar(ScalarGaussian, GibbsSampling):
    '''
    Conjugate normal prior on mean.
    '''
    def __repr__(self):
        return 'ScalarGaussianFixedvar(mu=%0.2f)' % (self.mu,)

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


class ScalarGaussianMaxLikelihood(ScalarGaussian):
    def max_likelihood(self,data,weights=None):
        data = flattendata(data)

        if weights is not None:
            weights = flattendata(weights)
        else:
            weights = np.ones(data.shape)

        weightsum = weights.sum()
        self.mu = np.dot(weights,data) / weightsum
        self.sigmasq = np.dot(weights,(data-self.mu)**2) / weightsum


##############
#  Discrete  #
##############

class Multinomial(GibbsSampling, MeanField):
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
    def __repr__(self):
        return 'Multinomial(weights=%s)' % (self.weights,)

    def __init__(self,weights=None,alpha_0=None,alphav_0=None,K=None):
        assert (isinstance(alphav_0,np.ndarray) and alphav_0.ndim == 1) ^ \
                (K is not None and alpha_0 is not None)

        if alphav_0 is not None:
            self.alphav_0 = alphav_0
            self.K = alphav_0.shape[0]
        else:
            self.K = K
            self.alphav_0 = np.ones(K)*alpha_0/K

        if weights is not None:
            self.weights = weights
        else:
            self.resample()
        self._alpha_mf = self.weights * self.alphav_0.sum()

    def rvs(self,size=None):
        return sample_discrete(self.weights,size)

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    def _posterior_hypparams(self,counts):
        return self.alphav_0 + counts

    ### Gibbs sampling

    def resample(self,data=[]):
        hypparams = self._posterior_hypparams(*self._get_statistics(data))
        self.weights = np.random.dirichlet(np.where(hypparams>1e-2,hypparams,1e-2))

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        K = self.K
        if isinstance(data,np.ndarray):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)
        return counts,

    ### Mean Field

    def meanfieldupdate(self,data,weights):
        # update
        self._alpha_mf = self._posterior_hypparams(*self._get_weighted_statistics(data,weights))
        self.weights = self._alpha_mf / self._alpha_mf.sum() # for plotting

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood(np.arange(self.K))
        q_entropy = -1* ((logpitilde*(self._alpha_mf-1)).sum() \
                + special.gammaln(self._alpha_mf.sum()) - special.gammaln(self._alpha_mf).sum())
        p_avgengy = special.gammaln(self.alphav_0.sum()) - special.gammaln(self.alphav_0).sum() \
                + ((self.alphav_0-1)*logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        # this may only make sense if np.all(x == np.arange(self.K))...
        return special.digamma(self._alpha_mf[x]) - special.digamma(self._alpha_mf.sum())

    def _get_weighted_statistics(self,data,weights):
        # data is just a placeholder; technically it should be
        # np.arange(self.K)[na,:].repeat(N,axis=0)
        assert isinstance(weights,np.ndarray) or \
                (isinstance(weights,list) and
                        all(isinstance(w,np.ndarray) for w in weights))

        if isinstance(data,np.ndarray):
            counts = weights.sum(0)
        else:
            counts = sum(w.sum(0) for w in weights)
        return counts,


class MultinomialConcentration(Multinomial):
    '''
    Multinomial with resampling of the symmetric Dirichlet concentration
    parameter.

        concentration ~ Gamma(a_0,b_0)

    The Dirichlet prior over pi is then

        pi ~ Dir(concentration/K)
    '''
    def __init__(self,a_0,b_0,K,concentration=None,weights=None):
        self.concentration = DirGamma(a_0=a_0,b_0=b_0,K=K,
                concentration=concentration)
        super(MultinomialConcentration,self).__init__(alpha_0=self.concentration.concentration,
                K=K,weights=weights)

    def resample(self,data=[],niter=10):
        if isinstance(data,list):
            counts = map(np.bincount,data)
        else:
            counts = np.bincount(data)

        for itr in range(niter):
            self.concentration.resample(counts,niter=1)
            self.alphav_0 = np.ones(self.K) * self.concentration.concentration
            super(MultinomialConcentration,self).resample(data)

    def meanfieldupdate(self,*args,**kwargs): # TODO
        warn('MeanField not implemented for %s; concentration parameter will stay fixed')
        super(MultinomialConcentration,self).meanfieldupdate(*args,**kwargs)


class Geometric(GibbsSampling, Collapsed):
    '''
    Geometric distribution with a conjugate beta prior.
    NOTE: the support is {1,2,3,...}

    Hyperparameters:
        alpha_0, beta_0

    Parameter is the success probability:
        p
    '''
    def __repr__(self):
        return 'Geometric(p=%0.2f)' % (self.p,)

    def __init__(self,alpha_0,beta_0,p=None):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        if p is not None:
            self.p = p
        else:
            self.resample()

    def _posterior_hypparams(self,n,tot):
        return self.alpha_0 + n, self.beta_0 + tot

    def log_likelihood(self,x):
        x = np.array(x,ndmin=1)
        raw = np.empty(x.shape)
        raw[x>0] = (x[x>0]-1.)*np.log(1.-self.p) + np.log(self.p)
        raw[x<1] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]

    def pmf(self,x):
        return stats.geom.pmf(x,self.p)

    def rvs(self,size=None):
        return np.random.geometric(self.p,size=size)

    ### Gibbs sampling

    def resample(self,data=[]):
        self.p = np.random.beta(*self._posterior_hypparams(*self._get_statistics(data)))

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum() - n
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data) - n
        else:
            assert isinstance(data,int)
            n = 1
            tot = data-1
        return n, tot

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.alpha_0,self.beta_0)

    def _log_partition_function(self,alpha,beta):
        return special.betaln(alpha,beta)


class Poisson(GibbsSampling, Collapsed):
    '''
    Poisson distribution with a conjugate Gamma prior.

    NOTE: the support is {0,1,2,...}

    Hyperparameters (following Wikipedia's notation):
        alpha_0, beta_0

    Parameter is the mean/variance parameter:
        lmbda
    '''
    def __repr__(self):
        return 'Poisson(lmbda=%0.2f)' % (self.lmbda,)

    def __init__(self,alpha_0,beta_0,lmbda=None):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.resample()

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

    ### Gibbs Sampling

    def resample(self,data=[]):
        alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        self.lmbda = np.random.gamma(alpha_n,1/beta_n)

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum()
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert isinstance(data,int)
            n = 1
            tot = data
        return n, tot

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.alpha_0,self.beta_0) \
                - self._get_sum_of_gammas(data)

    def _log_partition_function(self,alpha,beta):
        return special.gammaln(alpha) - alpha * np.log(beta)

    def _get_sum_of_gammas(self,data):
        if isinstance(data,np.ndarray):
            return special.gammaln(data+1).sum()
        elif isinstance(data,list):
            return sum(special.gammaln(d+1).sum() for d in data)
        else:
            assert isinstance(data,int)
            return special.gammaln(data+1)


class NegativeBinomial(GibbsSampling):
    '''
    Negative Binomial distribution with a conjugate beta prior on p and a
    separate gamma prior on r. The parameter r does not need to be an integer.
    If r is an integer, then x ~ NegBin(r,p) is the same as
    x = np.random.geometric(1-p,size=r).sum() - r
    where r is subtracted to make the geometric support be {0,1,2,...}
    Mean is r*p/(1-p), var is r*p/(1-p)**2

    Uses the data augemntation sampling method from Zhou et al. ICML 2012

    NOTE: the support is {0,1,2,...}.

    Hyperparameters:
        k_0, theta_0: r ~ Gamma(k, theta)
                      or r = np.random.gamma(k,theta)
        alpha_0, beta_0: p ~ Beta(alpha,beta)
                      or p = np.random.beta(alpha,beta)

    Parameters:
        r
        p
    '''
    def __repr__(self):
        return 'NegativeBinomial(r=%0.2f,p=%0.2f)' % (self.r,self.p)

    def __init__(self,k_0,theta_0,alpha_0,beta_0,r=None,p=None):
        self.k_0 = k_0
        self.theta_0 = theta_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if r is None or p is None:
            self.resample()
        else:
            self.r = r
            self.p = p

    def resample(self,data=[],niter=20):
        if getdatasize(data) == 0:
            self.p = np.random.beta(self.alpha_0,self.beta_0)
            self.r = np.random.gamma(self.k_0,self.theta_0)
        else:
            data = flattendata(data)
            N = len(data)
            for itr in range(niter):
                ### resample r
                msum = 0.
                for n in data:
                    msum += (np.random.rand(n) < self.r/(np.arange(n)+self.r)).sum()
                self.r = np.random.gamma(self.k_0 + self.msum, 1/(1/self.theta_0 - N*np.log(1-self.p)))
                ### resample p
                self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)

    def _resample_logseriesaug(self,data=[],niter=20):
        if getdatasize(data) == 0:
            self.p = np.random.beta(self.alpha_0,self.beta_0)
            self.r = np.random.gamma(self.k_0,self.theta_0)
        else:
            data = flattendata(data)
            N = data.shape[0]
            logF = self.logF
            L_i = np.zeros(N)
            data_nz = data[data > 0]
            for itr in range(niter):
                logR = np.arange(1,logF.shape[1]+1)*np.log(self.r) + logF
                L_i[data > 0] = sample_discrete_from_log(logR[data_nz-1,:data_nz.max()],axis=1)+1
                self.r = np.random.gamma(self.k_0 + L_i.sum(), 1/(1/self.theta_0 - np.log(1-self.p)*N))
                self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)

    def rvs(self,size=None):
        return np.random.poisson(np.random.gamma(self.r,self.p/(1-self.p),size=size))

    def log_likelihood(self,x):
        x = np.array(x,ndmin=1)
        xnn = x[x >= 0]
        raw = np.empty(x.shape)
        raw[x>=0] = special.gammaln(self.r + xnn) - special.gammaln(self.r) - special.gammaln(xnn+1)\
                + self.r*np.log(1-self.p) + xnn*np.log(self.p)
        raw[x<0] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]

    # @classmethod
    # def _set_up_logF(cls):
    #     if not hasattr(cls,'logF'):
    #         # actually indexes logF[0,0] to correspond to log(F(1,1)) in Zhou
    #         # paper, but keeps track of that alignment with the other code!
    #         # especially arange(1,...), only using nonzero data and shifting it
    #         SIZE = 500

    #         logF = -np.inf * np.ones((SIZE,SIZE))
    #         logF[0,0] = 0.
    #         for m in range(1,logF.shape[0]):
    #             prevrow = np.exp(logF[m-1] - logF[m-1].max())
    #             logF[m] = np.log(np.convolve(prevrow,[0,m,1],'same')) + logF[m-1].max()
    #         cls.logF = logF


################################
#  Special Case Distributions  #
################################

# TODO maybe move these to another module? priors with funny likelihoods

class CRPGamma(GibbsSampling):
    '''
    Implements Gamma(a,b) prior over DP/CRP concentration parameter given
    CRP data (integrating out the weights). NOT for Gamma/Poisson, which would
    be called Poisson.
    see appendix A of http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf
    and appendix C of Emily Fox's PhD thesis
    the notation of w's and s's follows from the HDP paper
    '''
    def __repr__(self):
        return 'CRPGamma(concentration=%0.2f)' % self.concentration

    def __init__(self,a_0,b_0,concentration=None):
        self.a_0 = a_0
        self.b_0 = b_0

        if concentration is not None:
            self.concentration = concentration
        else:
            self.resample(niter=1)

    def log_likelihood(self,x):
        raise NotImplementedError, 'product of gammas' # TODO

    def rvs(self,customer_counts):
        '''
        Number of distinct tables. Not complete CRPs. customer_counts is a list
        of customer counts, and len(customer_counts) is the number of
        restaurants.
        '''
        assert isinstance(customer_counts,list) or isinstance(customer_counts,int)
        if isinstance(customer_counts,int):
            customer_counts = [customer_counts]

        restaurants = []
        for num in customer_counts:
            tables = []
            for c in range(num):
                newidx = sample_discrete(np.array(tables + [self.concentration]))
                if newidx == len(tables):
                    tables += [1]
                else:
                    tables[newidx] += 1
            restaurants.append(tables)
        return restaurants if len(restaurants) > 1 else restaurants[0]

    def resample(self,data=[],niter=30):
        for itr in range(niter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(data))
            self.concentration = np.random.gamma(a_n,scale=1./b_n)

    def _posterior_hypparams(self,sample_numbers,total_num_distinct):
        # NOTE: this is a stochastic function
        sample_numbers = np.array(sample_numbers)
        sample_numbers = sample_numbers[sample_numbers > 0]
        if total_num_distinct > 0:
            wvec = np.random.beta(self.concentration+1,sample_numbers)
            svec = np.array(stats.bernoulli.rvs(sample_numbers/(sample_numbers+self.concentration)))
            return self.a_0 + total_num_distinct-svec.sum(), (self.b_0 - np.log(wvec).sum())
        else:
            return self.a_0, self.b_0

    def _get_statistics(self,data):
        # data is a list of CRP samples, each of which is written as a list of
        # counts of customers at tables, i.e.
        # [5 7 2 ... 1 ]
        assert isinstance(data,list)
        if len(data) == 0:
            sample_numbers = 0
            total_num_distinct = 0
        else:
            if isinstance(data[0],list):
                sample_numbers = np.array(map(sum,data))
                total_num_distinct = sum(map(len,data))
            else:
                sample_numbers = np.array(sum(data))
                total_num_distinct = len(data)
        return sample_numbers, total_num_distinct


class DirGamma(CRPGamma):
    '''
    Implements a Gamma(a_0,b_0) prior over finite dirichlet concentration
    parameter. The concentration is scaled according to the weak-limit according
    to the number of dimensions K.

    For each set of counts i, the model is
        concentration ~ Gamma(a_0,b_0)
        pi_i ~ Dir(concentration/K)
        data_i ~ Multinomial(pi_i)
    '''
    def __repr__(self):
        return 'DirGamma(concentration=%0.2f/%d)' % (self.concentration*self.K,self.K)

    def __init__(self,K,a_0,b_0,concentration=None):
        self.K = K
        super(DirGamma,self).__init__(a_0=a_0,b_0=b_0,
                concentration=concentration)

    def rvs(self,sample_counts):
        if isinstance(sample_counts,int):
            sample_counts = [sample_counts]
        out = np.empty((len(sample_counts),self.K),dtype=int)
        for idx,c in enumerate(sample_counts):
            out[idx] = np.random.multinomial(c,
                np.random.dirichlet(self.concentration * np.ones(self.K)))
        return out if out.shape[0] > 1 else out[0]

    def resample(self,data=[],niter=50,weighted_cols=None):
        if weighted_cols is not None:
            self.weighted_cols = weighted_cols
        else:
            self.weighted_cols = np.ones(self.K)

        if getdatasize(data) > 0:
            for itr in range(niter):
                super(DirGamma,self).resample(data,niter=1)
                self.concentration /= self.K
        else:
            super(DirGamma,self).resample(data,niter=1)
            self.concentration /= self.K


    def _get_statistics(self,data):
        counts = np.array(data,ndmin=2)

        # sample m's
        if counts.sum() == 0:
            return 0, 0
        else:
            m = 0
            for (i,j), n in np.ndenumerate(counts):
                m += (np.random.rand(n) < self.concentration*self.K*self.weighted_cols[j] \
                        / (np.arange(n)+self.concentration*self.K*self.weighted_cols[j])).sum()
            return counts.sum(1), m

