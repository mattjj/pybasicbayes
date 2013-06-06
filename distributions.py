from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
from numpy.core.umath_tests import inner1d
import scipy.weave
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import abc
from warnings import warn

from abstractions import Distribution, GibbsSampling,\
        MeanField, Collapsed, MaxLikelihood
from util.stats import sample_niw, sample_invwishart, invwishart_entropy,\
        invwishart_log_partitionfunction, sample_discrete,\
        sample_discrete_from_log, getdatasize, flattendata,\
        getdatadimension, combinedata, multivariate_t_loglik
import util.general

# TODO reduce reallocation of parameters
# TODO fix up docstrings to work with base classes

################
#  Continuous  #
################

class _GaussianBase(Distribution):
    __metaclass__ = abc.ABCMeta

    ### distribution stuff

    def rvs(self,size=None):
        return np.random.multivariate_normal(mean=self.mu,cov=self.sigma,size=size)

    def log_likelihood(self,x):
        assert x.dtype == np.float64
        mu, sigma, D = self.mu, self.sigma, self.D
        x = np.reshape(x,(-1,D)) - mu
        xs,LT = util.general.solve_chofactor_system(sigma,x.T,overwrite_b=True)
        return -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) - np.log(LT.diagonal()).sum()

    ### plotting

    def plot(self,data=None,color='b',plot_params=True,label=''):
        from util.plot import project_data, plot_gaussian_projection, plot_gaussian_2D
        if data is not None:
            data = flattendata(data)

        if self.D > 2 and ((not hasattr(self,'plotting_subspace_basis'))
                or (self.plotting_subspace_basis.shape[1] != self.D)):
            # TODO improve this bookkeeping. need a notion of collection

            subspace = np.random.randn(self.D,2)
            self.__class__.plotting_subspace_basis = np.linalg.qr(subspace)[0].T.copy()

        if data is not None:
            if self.D > 2:
                data = project_data(data,self.plotting_subspace_basis)
            plt.plot(data[:,0],data[:,1],marker='.',linestyle=' ',color=color)

        if plot_params:
            if self.D > 2:
                plot_gaussian_projection(self.mu,self.sigma,self.plotting_subspace_basis,
                        color=color,label=label)
            else:
                plot_gaussian_2D(self.mu,self.sigma,color=color,label=label)

    # TODO improve this
    def to_json_dict(self):
        assert self.D == 2
        U,s,_ = np.linalg.svd(self.sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),'ry':np.sqrt(s[1]),'theta':theta}


class Gaussian(_GaussianBase, GibbsSampling, MeanField, Collapsed, MaxLikelihood):
    '''
    Multivariate Gaussian distribution class.

    NOTE: Only works for 2 or more dimensions. For a scalar Gaussian, use one of
    the scalar classes.  Uses a conjugate Normal/Inverse-Wishart prior.

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

    def num_parameters(self):
        return self.D*(self.D+1)/2

    @staticmethod
    def _get_statistics(data,D):
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

    @staticmethod
    def _get_weighted_statistics(data,weights,D):
        # NOTE: _get_statistics is special case with all weights being 1
        # this is kept as a separate method for speed and modularity
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
                sample_niw(*self._posterior_hypparams(*self._get_statistics(data,self.D)))

    ### Mean Field

    # NOTE my sumsq is Bishop's Nk*Sk

    def meanfieldupdate(self,data,weights):
        # update
        self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights,self.D))
        self._sigma_mf_chol = None
        self.mu, self.sigma = self._mu_mf, self._sigma_mf/(self._nu_mf - self.D - 1) # for plotting

    def _get_sigma_mf_chol(self):
        if not hasattr(self,'_sigma_mf_chol') or self._sigma_mf_chol is None:
            self._sigma_mf_chol = util.general.cholesky(self._sigma_mf)
        return self._sigma_mf_chol

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        D = self.D
        loglmbdatilde = self._loglmbdatilde()
        chol = self._get_sigma_mf_chol()

        # see Eq. 10.77 in Bishop
        q_entropy = -0.5 * (loglmbdatilde + self.D * (np.log(self._kappa_mf/(2*np.pi))-1)) \
                + invwishart_entropy(self._sigma_mf,self._nu_mf,chol)
        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde \
                - D*self.kappa_0/self._kappa_mf - self.kappa_0*self._nu_mf*\
                np.dot(self._mu_mf -
                    self.mu_0,util.general.solve_psd(self._sigma_mf,self._mu_mf - self.mu_0,chol=chol))) \
                + invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
                + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self._nu_mf*\
                util.general.solve_psd(self._sigma_mf,self.sigma_0,chol=chol).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self._mu_mf, self._sigma_mf, self._kappa_mf, self._nu_mf
        D = self.D
        x = np.reshape(x,(-1,D)) - mu_n # x is now centered
        chol = self._get_sigma_mf_chol()
        xs = util.general.solve_triangular(chol,x.T,overwrite_b=True)

        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi)

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        chol = self._get_sigma_mf_chol()
        return special.digamma((self._nu_mf-np.arange(self.D))/2).sum() \
                + self.D*np.log(2) - 2*np.log(chol.diagonal()).sum()

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n, D = getdatasize(data), self.D
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data,self.D))) \
                - self._log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
                - n*D/2 * np.log(2*np.pi)

    def _log_partition_function(self,mu,sigma,kappa,nu):
        D = self.D
        chol = util.general.cholesky(sigma)
        return nu*D/2*np.log(2) + special.multigammaln(nu/2,D) + D/2*np.log(2*np.pi/kappa) \
                - nu*np.log(chol.diagonal()).sum()

    def log_predictive_studentt_datapoints(self,datapoints,olddata):
        mu_n, sigma_n, kappa_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata,self.D))
        D = self.D
        return multivariate_t_loglik(datapoints,nu_n-D+1,mu_n,(kappa_n+1)/(kappa_n*(nu_n-D+1))*sigma_n)

    def log_predictive_studentt(self,newdata,olddata):
        # an alternative computation to the generic log_predictive, which is implemented
        # in terms of log_marginal_likelihood. mostly for testing, I think
        newdata = np.atleast_2d(newdata)
        return sum(self.log_predictive_studentt_datapoints(d,combinedata((olddata,newdata[:i])))[0]
                        for i,d in enumerate(newdata))

    ### Max likelihood

    # NOTE: could also use sumsq/(n-1) as the covariance estimate, which would
    # be unbiased but not max likelihood, but if we're in the regime where that
    # matters we've got bigger problems!

    def max_likelihood(self,data,weights=None):
        D = self.D
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data,D)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights,D)

        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n < D or (np.linalg.svd(sumsq,compute_uv=False) > 1e-6).sum() < D:
            # broken!
            self.mu = 99999999*np.ones(D)
            self.sigma = np.eye(D)
            self.broken = True
        else:
            self.mu = muhat
            self.sigma = sumsq/n

    @classmethod
    def max_likelihood_constructor(cls,data,weights=None):
        D = getdatadimension(data)
        if weights is None:
            n, muhat, sumsq = cls._get_statistics(data,D)
        else:
            n, muhat, sumsq = cls._get_weighted_statistics(data,weights,D)
        assert n >= D

        return cls(muhat,sumsq/n,n,n,mu=muhat,sigma=sumsq/n)

    def max_a_posteriori(self,data,weights=None):
        # max likelihood with prior pseudocounts included in data
        D = self.D
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data,D)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights,D)

        self.mu, self.sigma, _, _ = self._posterior_hypparams(n,muhat,sumsq)


class GaussianFixedMean(_GaussianBase, GibbsSampling, MaxLikelihood):
    def __init__(self,mu,kappa_0,sigma_0,sigma=None):
        self.mu = mu
        self.kappa_0 = kappa_0
        self.sigma_0 = sigma_0
        self.D = mu.shape[0]

        if sigma is None:
            self.resample()
        else:
            self.sigma = sigma

    def num_parameters(self):
        return self.D*(self.D+1)/2

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                centered = data - self.mu
                sumsq = centered.T.dot(centered)
            else:
                sumsq = sum((d-self.mu).T.dot(d-self.mu) for d in data)
        else:
            sumsq = None
        return n, sumsq

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > 0:
                centered = data - self.mu
                sumsq = centered.T.dot(weights[:,na]*centered)
            else:
                sumsq = None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                sumsq = sum((d-self.mu).T.dot(w[:,na]*(d-self.mu)) for w,d in zip(weights,data)) / neff
            else:
                sumsq = None

        return neff, sumsq

    def _posterior_hypparams(self,n,sumsq):
        kappa_0, sigma_0 = self.kappa_0, self.sigma_0
        if n > 0:
            kappa_n = kappa_0 + n
            sigma_n = self.sigma_0 + sumsq
            return sigma_n, kappa_n
        else:
            return sigma_0, kappa_0

    ### Gibbs sampling

    def resample(self,data=[]):
        self.sigma = sample_invwishart(*self._posterior_hypparams(*self._get_statistics(data)))

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, sumsq = self._get_statistics(data)
        else:
            n, sumsq = self._get_weighted_statistics(data,weights)

        D = self.D
        if n < D or (np.linalg.svd(sumsq,compute_uv=False) > 1e-6).sum() < D:
            # broken!
            self.sigma = np.eye(D)*1e-9
            self.broken = True
        else:
            self.sigma = sumsq/n


class GaussianFixedCov(_GaussianBase, GibbsSampling, MaxLikelihood):
    # See Gelman's Bayesian Data Analysis notation around Eq. 3.18, p. 85 in 2nd
    # Edition
    def __init__(self,sigma,mu_0,lmbda_0,mu=None):
        self.mu_0 = mu_0
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma)
        self.lmbda_inv_0 = np.linalg.inv(lmbda_0)
        self.D = self.mu_0.shape[0]

        if mu is None:
            self.resample()
        else:
            self.mu = mu

    def num_parameters(self):
        return self.D

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = data.mean(0)
            else:
                xbar = sum(d.sum(0) for d in data) / n
        else:
            xbar = None

        return n, xbar

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > 0:
                xbar = weights.dot(data) / neff
            else:
                xbar = None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                xbar = sum(w.dot(d) for w,d in zip(weights,data)) / neff
            else:
                xbar = None

        return neff, xbar

    def _posterior_hypparams(self,n,xbar):
        sigma_inv, mu_0, lmbda_inv_0 = self.sigma_inv, self.mu_0, self.lmbda_inv_0
        if n > 0:
            lmbda_inv_n = n*sigma_inv + lmbda_inv_0
            mu_n = np.linalg.solve(lmbda_inv_n, lmbda_inv_0.dot(mu_0) + n*sigma_inv.dot(xbar))
            return mu_n, lmbda_inv_n
        else:
            return mu_0, lmbda_inv_0

    ### Gibbs sampling

    def resample(self,data=[]):
        mu_n, sigma_n_inv = self._posterior_hypparams(*self._get_statistics(data))
        self.mu = util.general.solve_chofactor_system(sigma_n_inv,np.random.normal(size=self.D))[0] + mu_n

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, xbar = self._get_statistics(data)
        else:
            n, xbar = self._get_weighted_statistics(data,weights)

        self.mu = xbar


class GaussianNonConj(_GaussianBase, GibbsSampling):
    def __init__(self,mu_0,mu_sigma_0,kappa_0,sigma_sigma_0,mu=None,sigma=None):
        self._sigma_distn = GaussianFixedMean(mu_0,kappa_0,sigma_sigma_0,sigma)
        self._mu_distn = GaussianFixedCov(self._sigma_distn.sigma,mu_0,mu_sigma_0,mu)
        self.D = mu_0.shape[0]

    @property
    def mu(self):
        return self._mu_distn.mu

    @property
    def sigma(self):
        return self._sigma_distn.sigma

    ### Gibbs sampling

    def resample(self,data=[],niter=30):
        if getdatasize(data) == 0:
            niter = 1

        for itr in xrange(niter):
            # resample mu
            self._mu_distn.sigma = self._sigma_distn.sigma
            self._mu_distn.resample(data)

            # resample sigma
            self._sigma_distn.mu = self._mu_distn.mu
            self._sigma_distn.resample(data)


# TODO collapsed, meanfield, max_likelihood
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
        D = self.D = mu_0.shape[0]
        self.mu_0 = mu_0

        # all the s's refer to the fact that these are vectors of length
        # len(mu_0) OR scalars
        if isinstance(nus_0,int) or isinstance(nus_0,float):
            nus_0 = nus_0*np.ones(D)
        if isinstance(alphas_0,int) or isinstance(alphas_0,float):
            alphas_0 = alphas_0*np.ones(D)
        if isinstance(betas_0,int) or isinstance(betas_0,float):
            betas_0 = betas_0*np.ones(D)

        self.nus_0 = nus_0
        self.alphas_0 = alphas_0
        self.betas_0 = betas_0

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
        D = self.D
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                data = np.reshape(data,(-1,D))
                xbar = data.mean(0)
                centered = data - xbar
                sumsq = np.diag(np.dot(centered.T,centered))
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = np.diag(sum(((np.reshape(d,(-1,D)) - xbar)**2).sum(0) for d in data))
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq


# TODO collapsed, meanfield, max_likelihood
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
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).ravel()

    def __repr__(self):
        return self.__class__.__name__ + '(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

    def plot(self,data=None,color='b',plot_params=True):
        raise NotImplementedError # TODO


# TODO meanfield, max_likelihood
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
    def __init__(self,mu_0,tausq_0,sigmasq_0,nu_0,niter=20,
            mu=None,sigmasq=None,mubin=None,sigmasqbin=None):
        self.mu_0, self.tausq_0 = mu_0, tausq_0
        self.sigmasq_0, self.nu_0 = sigmasq_0, nu_0

        self.niter = niter

        self.mubin = mubin
        self.sigmasqbin = sigmasqbin

        if mu is None or sigmasq is None:
            self.resample()
        else:
            self.mu = mu
            self.sigmasq = sigmasq
            if mubin is not None and sigmasqbin is not None:
                self.mubin[...] = mu
                self.sigmasqbin[...] = sigmasq

    def resample(self,data=[],niter=None):
        n = getdatasize(data)
        niter = self.niter if niter is None else niter
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

##############
#  Discrete  #
##############

class Categorical(GibbsSampling, MeanField, MaxLikelihood):
    '''
    This class represents a categorical distribution over labels, where the
    parameter is weights and the prior is a Dirichlet distribution.
    For example, if K == 3, then five samples may look like
        [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls. In other
    words, generated data or data passed to log_likelihood are indices, not
    indicator variables!  (But when 'weighted data' is passed, like in mean
    field or weighted max likelihood, the weights are over indicator
    variables...)

    This class can be used as a weak limit approximation for a DP, particularly by
    calling __init__ with alpha_0 and K arguments, in which case the prior will be
    a symmetric Dirichlet with K components and parameter alpha_0/K; K is then the
    weak limit approximation parameter.

    Hyperparaemters:
        alphav_0 (vector) OR alpha_0 (scalar) and K

    Parameters:
        weights, a vector encoding a finite pmf
    '''
    def __repr__(self):
        return '%s(weights=%s)' % (self.__class__.__name__,self.weights)

    def __init__(self,weights=None,alpha_0=None,K=None,alphav_0=None):
        assert (isinstance(alphav_0,np.ndarray) and alphav_0.ndim == 1) ^ \
                (K is not None and alpha_0 is not None)

        if alphav_0 is not None:
            self.alphav_0 = alphav_0
            self.K = alphav_0.shape[0]
        else:
            self.K = K
            self.alphav_0 = np.repeat(alpha_0/K,K)

        if weights is not None:
            self.weights = weights
        else:
            self.resample()
        self._alpha_mf = self.weights * self.alphav_0.sum()

    def num_parameters(self):
        return self.K

    def rvs(self,size=None):
        return sample_discrete(self.weights,size)

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    def _posterior_hypparams(self,counts):
        return self.alphav_0 + counts

    ### Gibbs sampling

    def resample(self,data=[]):
        'data is an array of indices (i.e. labels) or a list of such arrays'
        hypparams = self._posterior_hypparams(*self._get_statistics(data,self.K))
        self.weights = np.random.dirichlet(np.where(hypparams>1e-2,hypparams,1e-2))

    @staticmethod
    def _get_statistics(data,K):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

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

    @staticmethod
    def _get_weighted_statistics(data,weights):
        # data is just a placeholder; technically it should always be
        # np.arange(K)[na,:].repeat(N,axis=0)
        assert isinstance(weights,np.ndarray) or \
                (isinstance(weights,list) and all(isinstance(w,np.ndarray) for w in weights))

        if isinstance(weights,np.ndarray):
            counts = weights.sum(0)
        else:
            counts = sum(w.sum(0) for w in weights)
        return counts,

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)

        self.weights = counts/counts.sum()

    def max_likelihood_withprior(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)

        self.weights = counts/counts.sum()


class CategoricalAndConcentration(Categorical):
    '''
    Categorical with resampling of the symmetric Dirichlet concentration
    parameter.

        concentration ~ Gamma(a_0,b_0)

    The Dirichlet prior over pi is then

        pi ~ Dir(concentration/K)
    '''
    def __init__(self,a_0,b_0,K,concentration=None,weights=None):
        self.concentration = DirGamma(a_0=a_0,b_0=b_0,K=K,concentration=concentration)
        super(CategoricalAndConcentration,self).__init__(alpha_0=self.concentration.concentration,
                K=K,weights=weights)

    def resample(self,data=[]):
        counts, = self._get_statistics(data,self.K)
        self.concentration.resample(counts)
        self.alphav_0 = np.repeat(self.concentration.concentration/self.K,self.K)
        super(CategoricalAndConcentration,self).resample(data)

    def resample_just_weights(self,data=[]):
        super(CategoricalAndConcentration,self).resample(data)

    def meanfieldupdate(self,*args,**kwargs): # TODO
        warn('MeanField not implemented for %s; concentration parameter will stay fixed')
        super(CategoricalAndConcentration,self).meanfieldupdate(*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError, "max_likelihood doesn't make sense on this object"


class Multinomial(Categorical):
    '''
    Similar to Categorical, but data are counts.

    For example, if K == 3, then a sample with n=5 might be
        [2,2,1]

    A Poisson process conditioned on the number of points emitted.
    '''
    def log_likelihood(self,x):
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.K
        return (x*np.log(self.weights)).sum(1) \
                + special.gammaln(x.sum(1)+1)[:,na] - special.gammaln(x+1).sum(1)[:,na]

    def resample(self,data=[]):
        'data is an array of counts or a list of such arrays)'
        warn('untested')
        super(Multinomial,self).resample(data)

    @staticmethod
    def _get_statistics(data,K):
        warn('untested')
        if isinstance(data,np.ndarray):
            return data,
        else:
            data = np.array(data)
            if data.size == 0:
                return np.zeros(K,dtype=int),
            return np.array(data).sum(0),

    @staticmethod
    def _get_weighted_statistics(data,weights):
        raise NotImplementedError # TODO

    def max_likelihood(self,counts,weights=None):
        warn('untested')
        if weights is None:
            self.weights = counts /counts.sum()
        else:
            raise NotImplementedError # TODO


# TODO this is all repeated code from CategoricalAndConcentration!
# metaprogramming, please!
class MultinomialAndConcentration(Multinomial):
    '''
    Similar to CategoricalAndConcentration, but data are counts.
    '''
    def __init__(self,a_0,b_0,K,concentration=None,weights=None):
        self.concentration = DirGamma(a_0=a_0,b_0=b_0,K=K,concentration=concentration)
        super(MultinomialAndConcentration,self).__init__(alpha_0=self.concentration.concentration,K=K,weights=weights)

    def resample(self,data=[]):
        warn('untested')
        counts, = self._get_statistics(data,self.K)
        self.concentration.resample(counts)
        self.alphav_0 = np.repeat(self.concentration.concentration/self.K,self.K)
        super(MultinomialAndConcentration,self).resample(data)

    def meanfieldupdate(self,*args,**kwargs): # TODO
        warn('MeanField not implemented for %s; concentration parameter will stay fixed')
        super(MultinomialAndConcentration,self).meanfieldupdate(*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError, "max_likelihood doesn't make sense on this object"


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
        return '%s(p=%0.2f)' % (self.__class__.__name__,self.p,)

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

    def log_sf(self,x):
        return stats.geom.logsf(x,self.p)

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

    def log_sf(self,x):
        return stats.poisson.logsf(x,self.lmbda)

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

    def _get_weighted_statistics(self,data,weights):
        pass # TODO

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

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        self.lmbda = tot/n


class _NegativeBinomialBase(Distribution):
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
        return '%s(r=%0.2f,p=%0.2f)' % (self.__class__.__name__,self.r,self.p)

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

    def log_likelihood(self,x,r=None,p=None):
        if r is None or p is None:
            r,p = self.r, self.p
        x = np.array(x,ndmin=1)

        if self.p > 0:
            xnn = x[x >= 0]
            raw = np.empty(x.shape)
            raw[x>=0] = special.gammaln(r + xnn) - special.gammaln(r) - special.gammaln(xnn+1)\
                    + r*np.log(1-p) + xnn*np.log(p)
            raw[x<0] = -np.inf
            return raw if isinstance(x,np.ndarray) else raw[0]
        else:
            raw = np.log(np.zeros(x.shape))
            raw[x == 0] = 0.
            return raw if isinstance(x,np.ndarray) else raw[0]

    def log_sf(self,x):
        scalar = not isinstance(x,np.ndarray)
        x = np.atleast_1d(x)
        ret = np.log(special.betainc(x+1,self.r,self.p))
        ret[x < 0] = np.log(1.)
        if scalar:
            return ret[0]
        else:
            return ret

    def rvs(self,size=None):
        return np.random.poisson(np.random.gamma(self.r,self.p/(1-self.p),size=size))


class NegativeBinomial(_NegativeBinomialBase, GibbsSampling):
    def resample(self,data=[],niter=20):
        if getdatasize(data) == 0:
            self.p = np.random.beta(self.alpha_0,self.beta_0)
            self.r = np.random.gamma(self.k_0,self.theta_0)
        else:
            data = flattendata(data)
            N = len(data)
            for itr in range(niter):
                ### resample r
                msum = np.array(0.)
                r = self.r
                scipy.weave.inline(
                        '''
                        int tot = 0;
                        for (int i=0; i < N; i++) {
                            for (int j=0; j < data[i]; j++) {
                                tot += (((float) rand()) / RAND_MAX) < (((float) r)/(j+r));
                            }
                        }
                        *msum = tot;
                        ''',
                        ['N','data','r','msum'],
                        extra_compile_args=['-O3'])
                self.r = np.random.gamma(self.k_0 + msum, 1/(1/self.theta_0 - N*np.log(1-self.p)))
                ### resample p
                self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)

    def resample_python(self,data=[],niter=20):
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
                self.r = np.random.gamma(self.k_0 + msum, 1/(1/self.theta_0 - N*np.log(1-self.p)))
                ### resample p
                self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)

    ### OLD unused alternatives

    def resample_logseriesaug(self,data=[],niter=20):
        # an alternative algorithm, kind of opaque and no advantages...
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

    @classmethod
    def _set_up_logF(cls):
        if not hasattr(cls,'logF'):
            # actually indexes logF[0,0] to correspond to log(F(1,1)) in Zhou
            # paper, but keeps track of that alignment with the other code!
            # especially arange(1,...), only using nonzero data and shifting it
            SIZE = 500

            logF = -np.inf * np.ones((SIZE,SIZE))
            logF[0,0] = 0.
            for m in range(1,logF.shape[0]):
                prevrow = np.exp(logF[m-1] - logF[m-1].max())
                logF[m] = np.log(np.convolve(prevrow,[0,m,1],'same')) + logF[m-1].max()
            cls.logF = logF


class NegativeBinomialFixedR(_NegativeBinomialBase, GibbsSampling, MaxLikelihood):
    def __init__(self,r,alpha_0,beta_0,p=None):
        self.r = r
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if p is None:
            self.resample()
        else:
            self.p = p

    def resample(self,data=[]):
        if getdatasize(data) == 0:
            self.p = np.random.beta(self.alpha_0,self.beta_0)
        else:
            data = flattendata(data)
            N = len(data)
            self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)

    # TODO test
    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        self.p = (tot/n) / (self.r + tot/n)

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            assert np.all(data >= 0)
            data = np.atleast_1d(data)
            n, tot = data.shape[0], data.sum()
        else:
            assert all(np.all(d >= 0) for d in data)
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)

        return n, tot

    # TODO test
    def _get_weighted_statistics(self,data,weights):
        if isinstance(weights,np.ndarray):
            assert np.all(data >= 0)
            n, tot = weights.sum(), (data*weights).sum()
        else:
            assert all(np.all(d >= 0) for d in data)
            n = sum(w.sum() for w in weights)
            tot = sum((d*w).sum() for d,w in zip(data,weights))

        return n, tot

class NegativeBinomialIntegerR(NegativeBinomialFixedR, GibbsSampling, MaxLikelihood):
    '''
    Nonconjugate Discrete+Beta prior
    r_discrete_distribution is an array where index i is p(r=i+1)
    '''
    def __init__(self,r_discrete_distn,alpha_0,beta_0,r=None,p=None):
        self.r_discrete_distn = np.asarray(r_discrete_distn)
        self.r_discrete_distn /= self.r_discrete_distn.sum()
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if r is None or p is None:
            self.resample()
        else:
            self.r = r
            self.p = p

    def resample(self,data=[],nsteps=20):
        # MH / simulated annealing
        # see web.mit.edu/~wingated/www/introductions/mcmc-gibbs-intro.pdf
        if getdatasize(data) == 0:
            self.p = np.random.beta(self.alpha_0,self.beta_0)
            self.r = sample_discrete(self.r_discrete_distn)+1
        else:
            data = flattendata(data)
            N = len(data)

            current_log_prior_value = stats.beta.logpdf(self.p,self.alpha_0,self.beta_0) \
                    + np.log(self.r_discrete_distn[self.r-1])
            current_log_likelihood_value = np.sum(self.log_likelihood(data))

            for itr in xrange(nsteps):
                proposal_r = sample_discrete(self.r_discrete_distn)+1
                proposal_p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*proposal_r)

                proposal_log_prior_value =  stats.beta.logpdf(proposal_p,self.alpha_0,self.beta_0) \
                        + np.log(self.r_discrete_distn[self.r-1])
                proposal_log_likelihood_value = np.sum(self.log_likelihood(x=data,r=proposal_r,p=proposal_p))

                if np.isinf(proposal_log_likelihood_value) and np.isinf(current_log_likelihood_value):
                    accept_probability = 1.
                else:
                    accept_probability = np.exp(min(0.,
                        proposal_log_prior_value - current_log_prior_value \
                                + proposal_log_likelihood_value - current_log_likelihood_value))

                if np.random.rand() < accept_probability:
                    self.r, self.p = proposal_r, proposal_p
                    current_log_prior_value = proposal_log_prior_value
                    current_log_likelihood_value = proposal_log_likelihood_value

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        if n > 0:
            r_support = np.where(self.r_discrete_distn > 0)[0]+1
            rmin, rmax = r_support[0], r_support[-1]

            rs = np.arange(rmin,rmax+1)
            ps = (tot/n) / (rs + tot/n)

            # TODO make log_likelihood work with array args
            if isinstance(data,np.ndarray):
                likelihoods = np.array([self.log_likelihood(data,r=r,p=p).sum()
                                            for r,p in zip(rs,ps)])
            else:
                likelihoods = np.array([sum(self.log_likelihood(d,r=r,p=p).sum()
                                            for d in data) for r,p in zip(rs,ps)])

            self.r = rmin + likelihoods.argmax()
            self.p = ps[likelihoods.argmax()]

# class surgery, do you concur?
def _start_at_r(cls):
    class Wrapper(cls):
        def log_likelihood(self,x,*args,**kwargs):
            return super(Wrapper,self).log_likelihood(x-self.r,*args,**kwargs)

        def log_sf(self,x,*args,**kwargs):
            return super(Wrapper,self).log_sf(x-self.r)

        def rvs(self,size=None):
            return super(Wrapper,self).rvs(size)+self.r

        def resample(self,data=[],*args,**kwargs):
            if isinstance(data,np.ndarray):
                return super(Wrapper,self).resample(data-self.r,*args,**kwargs)
            else:
                return super(Wrapper,self).resample([d-self.r for d in data],*args,**kwargs)

        def max_likelihood(self,data,weights=None,*args,**kwargs):
            if weights is not None:
                raise NotImplementedError
            else:
                if isinstance(data,np.ndarray):
                    return super(Wrapper,self).max_likelihood(data-self.r,weights=None,*args,**kwargs)
                else:
                    return super(Wrapper,self).max_likelihood([d-self.r for d in data],weights=None,*args,**kwargs)

    Wrapper.__name__ = cls.__name__ + 'Variant'
    if cls.__doc__ is not None:
        Wrapper.__doc__ = 'Variant!\n\n' + cls.__doc__
    return Wrapper

NegativeBinomialVariant = _start_at_r(NegativeBinomial)
NegativeBinomialFixedRVariant = _start_at_r(NegativeBinomialFixedR)
NegativeBinomialIntegerRVariant = _start_at_r(NegativeBinomialIntegerR)

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
        raise NotImplementedError # TODO product of gammas

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

    def resample(self,data=[],niter=25):
        for itr in range(niter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(data))
            self.concentration = np.random.gamma(a_n,scale=1./b_n)

    def _posterior_hypparams(self,sample_numbers,total_num_distinct):
        # NOTE: this is a stochastic function: it samples auxiliary variables
        # and should be repeated
        if total_num_distinct > 0:
            sample_numbers = np.array(sample_numbers)
            sample_numbers = sample_numbers[sample_numbers > 0]

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

    @classmethod
    def test(cls,niter=50):
        d1 = cls(a_0=1.,b_0=1./4)
        data = d1.rvs([50 for i in range(50)])

        d2 = cls(a_0=1.,b_0=1./4)
        concs = []
        for itr in range(50):
            d2.resample(data,niter=niter)
            concs.append(d2.concentration)

        plt.hist(concs,normed=True)
        ymin, ymax = plt.ylim()
        plt.vlines([d1.concentration],ymin,ymax,linestyles='dashed')
        plt.ylim(ymin,ymax)
        plt.xlim(0,20)


class DirGamma(CRPGamma):
    '''
    Implements a Gamma(a_0,b_0) prior over finite dirichlet concentration
    parameter. The concentration is scaled according to the weak-limit according
    to the number of dimensions K. It stochastically splits counts into CRP
    counts and uses the CRPGamma class's methods.

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
                np.random.dirichlet(np.repeat(self.concentration/self.K,self.K)))
        return out if out.shape[0] > 1 else out[0]

    def resample(self,data=[],niter=25,weighted_cols=None):
        if weighted_cols is not None:
            self.weighted_cols = weighted_cols
        else:
            self.weighted_cols = np.ones(self.K)

        if isinstance(data,np.ndarray):
            size = data.sum()
        elif isinstance(data,list):
            size = sum(d.sum() for d in data)
        else:
            assert data == 0
            size = 0

        if size > 0:
            super(DirGamma,self).resample(data,niter=niter)
        else:
            super(DirGamma,self).resample(data,niter=1)

    def _get_statistics(self,data):
        counts = np.array(data,ndmin=2)

        # sample m's
        if counts.sum() == 0:
            return 0, 0
        else:
            msum = np.array(0.)
            weighted_cols = self.weighted_cols
            concentration = self.concentration
            N,K = counts.shape
            scipy.weave.inline(
                    '''
                    int tot = 0;
                    for (int i=0; i < N; i++) {
                        for (int j=0; j < K; j++) {
                            for (int c=0; c < counts[i*K + j]; c++) {
                                tot += ((float) rand()) / RAND_MAX <
                                    ((float) concentration/K*weighted_cols[j]) /
                                            (c + concentration/K*weighted_cols[j]);
                            }
                        }
                    }
                    *msum = tot;
                    ''',
                    ['weighted_cols','concentration','N','K','msum','counts'],
                    extra_compile_args=['-O3'])
            return counts.sum(1), int(msum)

    def _get_statistics_python(self,data):
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

    @classmethod
    def test(cls,niter=50):
        d1 = cls(K=25,a_0=1.,b_0=1./4)
        data = d1.rvs([50 for i in range(50)])

        d2 = cls(K=25,a_0=1.,b_0=1./4)
        concs = []
        for itr in range(50):
            d2.resample(data,niter=niter)
            concs.append(d2.concentration)

        plt.hist(concs,normed=True)
        ymin, ymax = plt.ylim()
        plt.vlines([d1.concentration],ymin,ymax,linestyles='dashed')
        plt.ylim(ymin,ymax)
        plt.xlim(0,20)

