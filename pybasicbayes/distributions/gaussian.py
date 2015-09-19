from __future__ import division
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
__all__ = \
    ['Gaussian', 'GaussianFixedMean', 'GaussianFixedCov', 'GaussianFixed',
     'GaussianNonConj', 'DiagonalGaussian', 'DiagonalGaussianNonconjNIG',
     'IsotropicGaussian', 'ScalarGaussianNIX', 'ScalarGaussianNonconjNIX',
     'ScalarGaussianNonconjNIG', 'ScalarGaussianFixedvar']

import numpy as np
from numpy import newaxis as na
from numpy.core.umath_tests import inner1d
import scipy.linalg
import scipy.stats as stats
import scipy.special as special
import copy

from pybasicbayes.abstractions import GibbsSampling, MeanField, \
    MeanFieldSVI, Collapsed, MaxLikelihood, MAP, Tempering
from pybasicbayes.distributions.meta import _FixedParamsMixin
from pybasicbayes.util.stats import sample_niw, invwishart_entropy, \
    sample_invwishart, invwishart_log_partitionfunction, \
    getdatasize, flattendata, getdatadimension, \
    combinedata, multivariate_t_loglik, gi, niw_expectedstats

weps = 1e-12


class _GaussianBase(object):
    @property
    def params(self):
        return dict(mu=self.mu, sigma=self.sigma)

    @property
    def D(self):
        return self.mu.shape[0]

    ### internals

    def getsigma(self):
        return self._sigma

    def setsigma(self,sigma):
        self._sigma = sigma
        self._sigma_chol = None

    sigma = property(getsigma,setsigma)

    @property
    def sigma_chol(self):
        if not hasattr(self,'_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    ### distribution stuff

    def rvs(self,size=None):
        size = 1 if size is None else size
        size = size + (self.mu.shape[0],) if isinstance(size,tuple) \
            else (size,self.mu.shape[0])
        return self.mu + np.random.normal(size=size).dot(self.sigma_chol.T)

    def log_likelihood(self,x):
        try:
            mu, D = self.mu, self.D
            sigma_chol = self.sigma_chol
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x).reshape((-1,D)) - mu
            xs = scipy.linalg.solve_triangular(sigma_chol,x.T,lower=True)
            out = -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) \
                - np.log(sigma_chol.diagonal()).sum()
            out[bads] = 0
            return out
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density
            return np.repeat(-np.inf,x.shape[0])

    ### plotting

    # TODO making animations, this seems to generate an extra notebook figure

    _scatterplot = None
    _parameterplot = None

    def plot(self,ax=None,data=None,indices=None,color='b',
             plot_params=True,label='',alpha=1.,
             update=False,draw=True):
        import matplotlib.pyplot as plt
        from pybasicbayes.util.plot import project_data, \
                plot_gaussian_projection, plot_gaussian_2D
        ax = ax if ax else plt.gca()
        D = self.D
        if data is not None:
            data = flattendata(data)

        if data is not None:
            if D > 2:
                plot_basis = np.random.RandomState(seed=0).randn(2,D)
                data = project_data(data,plot_basis)
            if update and self._scatterplot is not None:
                self._scatterplot.set_offsets(data)
                self._scatterplot.set_color(color)
            else:
                self._scatterplot = ax.scatter(
                    data[:,0],data[:,1],marker='.',color=color)

        if plot_params:
            if D > 2:
                plot_basis = np.random.RandomState(seed=0).randn(2,D)
                self._parameterplot = \
                    plot_gaussian_projection(
                        self.mu,self.sigma,plot_basis,
                        color=color,label=label,alpha=min(1-1e-3,alpha),
                        ax=ax, artists=self._parameterplot if update else None)
            else:
                self._parameterplot = \
                    plot_gaussian_2D(
                        self.mu,self.sigma,color=color,label=label,
                        alpha=min(1-1e-3,alpha), ax=ax,
                        artists=self._parameterplot if update else None)

        if draw:
            plt.draw()

        return [self._scatterplot] + list(self._parameterplot)

    def to_json_dict(self):
        D = self.mu.shape[0]
        assert D == 2
        U,s,_ = np.linalg.svd(self.sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),
                'ry':np.sqrt(s[1]), 'theta':theta}


class Gaussian(
        _GaussianBase, GibbsSampling, MeanField, MeanFieldSVI,
        Collapsed, MAP, MaxLikelihood):
    '''
    Multivariate Gaussian distribution class.

    NOTE: Only works for 2 or more dimensions. For a scalar Gaussian, use a
    scalar class.  Uses a conjugate Normal/Inverse-Wishart prior.

    Hyperparameters mostly follow Gelman et al.'s notation in Bayesian Data
    Analysis:
        nu_0, sigma_0, mu_0, kappa_0

    Parameters are mean and covariance matrix:
        mu, sigma
    '''

    def __init__(
            self, mu=None, sigma=None,
            mu_0=None, sigma_0=None, kappa_0=None, nu_0=None):
        self.mu = mu
        self.sigma = sigma

        self.mu_0    = self.mu_mf    = mu_0
        self.sigma_0 = self.sigma_mf = sigma_0
        self.kappa_0 = self.kappa_mf = kappa_0
        self.nu_0    = self.nu_mf    = nu_0

        # NOTE: resampling will set mu_mf and sigma_mf if necessary
        if mu is sigma is None \
                and not any(_ is None for _ in (mu_0,sigma_0,kappa_0,nu_0)):
            self.resample()  # initialize from prior
        if mu is not None and sigma is not None \
                and not any(_ is None for _ in (mu_0,sigma_0,kappa_0,nu_0)):
            self.mu_mf = mu
            self.sigma_mf = sigma * (self.nu_0 - self.mu_mf.shape[0] - 1)

    @property
    def hypparams(self):
        return dict(
            mu_0=self.mu_0,sigma_0=self.sigma_0,
            kappa_0=self.kappa_0,nu_0=self.nu_0)

    @property
    def natural_hypparam(self):
        return self._standard_to_natural(
            self.mu_0,self.sigma_0,self.kappa_0,self.nu_0)

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.mu_0, self.sigma_0, self.kappa_0, self.nu_0 = \
            self._natural_to_standard(natparam)

    def _standard_to_natural(self,mu_mf,sigma_mf,kappa_mf,nu_mf):
        D = sigma_mf.shape[0]
        out = np.zeros((D+2,D+2))
        out[:D,:D] = sigma_mf + kappa_mf * np.outer(mu_mf,mu_mf)
        out[:D,-2] = out[-2,:D] = kappa_mf * mu_mf
        out[-2,-2] = kappa_mf
        out[-1,-1] = nu_mf + 2 + D
        return out

    def _natural_to_standard(self,natparam):
        D = natparam.shape[0]-2
        A = natparam[:D,:D]
        b = natparam[:D,-2]
        c = natparam[-2,-2]
        d = natparam[-1,-1]
        return b/c, A - np.outer(b,b)/c, c, d - 2 - D

    @property
    def num_parameters(self):
        D = self.D
        return D*(D+1)/2

    @property
    def D(self):
        if self.mu is not None:
            return self.mu.shape[0]
        elif self.mu_0 is not None:
            return self.mu_0.shape[0]

    def _get_statistics(self,data,D=None):
        if D is None:
            D = self.D if self.D is not None else getdatadimension(data)
        out = np.zeros((D+2,D+2))
        if isinstance(data,np.ndarray):
            out[:D,:D] = data.T.dot(data)
            out[-2,:D] = out[:D,-2] = data.sum(0)
            out[-2,-2] = out[-1,-1] = data.shape[0]
            return out
        else:
            return sum(list(map(self._get_statistics,data)),out)

    def _get_weighted_statistics(self,data,weights,D=None):
        D = getdatadimension(data) if D is None else D
        out = np.zeros((D+2,D+2))
        if isinstance(data,np.ndarray):
            out[:D,:D] = data.T.dot(weights[:,na]*data)
            out[-2,:D] = out[:D,-2] = weights.dot(data)
            out[-2,-2] = out[-1,-1] = weights.sum()
            return out
        else:
            return sum(list(map(self._get_weighted_statistics,data,weights)),out)

    def _get_empty_statistics(self, D):
        out = np.zeros((D+2,D+2))
        return out

    def empirical_bayes(self,data):
        self.natural_hypparam = self._get_statistics(data)
        self.resample()  # intialize from prior given new hyperparameters
        return self

    @staticmethod
    def _stats_ensure_array(stats):
        if isinstance(stats, np.ndarray):
            return stats
        x, xxT, n = stats
        D = x.shape[-1]
        out = np.zeros((D+2,D+2))
        out[:D,:D] = xxT
        out[-2,:D] = out[:D,-2] = x
        out[-2,-2] = out[-1,-1] = n
        return out

    ### Gibbs sampling

    def resample(self,data=[]):
        D = len(self.mu_0)
        self.mu, self.sigma = \
            sample_niw(*self._natural_to_standard(
                self.natural_hypparam + self._get_statistics(data,D)))
        # NOTE: next lines let Gibbs sampling initialize mean
        nu = self.nu_mf if hasattr(self,'nu_mf') and self.nu_mf \
            else self.nu_0
        self.mu_mf, self._sigma_mf = self.mu, self.sigma * (nu - D - 1)
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new

    ### Mean Field

    def _resample_from_mf(self):
        self.mu, self.sigma = \
            sample_niw(*self._natural_to_standard(
                self.mf_natural_hypparam))
        return self

    def meanfieldupdate(self, data=None, weights=None, stats=None):
        assert (data is not None and weights is not None) ^ (stats is not None)
        stats = self._stats_ensure_array(stats) if stats is not None else \
            self._get_weighted_statistics(data, weights, self.mu_0.shape[0])
        self.mf_natural_hypparam = \
            self.natural_hypparam + stats

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        D = len(self.mu_0)
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                self.natural_hypparam
                + 1./prob
                * self._get_weighted_statistics(data,weights,D))

    @property
    def mf_natural_hypparam(self):
        return self._standard_to_natural(
            self.mu_mf,self.sigma_mf,self.kappa_mf,self.nu_mf)

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.mu_mf, self.sigma_mf, self.kappa_mf, self.nu_mf = \
            self._natural_to_standard(natparam)
        # NOTE: next line is for plotting
        self.mu, self.sigma = \
            self.mu_mf, self.sigma_mf/(self.nu_mf - self.mu_mf.shape[0] - 1)

    @property
    def sigma_mf(self):
        return self._sigma_mf

    @sigma_mf.setter
    def sigma_mf(self,val):
        self._sigma_mf = val
        self._sigma_mf_chol = None

    @property
    def sigma_mf_chol(self):
        if self._sigma_mf_chol is None:
            self._sigma_mf_chol = np.linalg.cholesky(self.sigma_mf)
        return self._sigma_mf_chol

    def get_vlb(self):
        D = len(self.mu_0)
        loglmbdatilde = self._loglmbdatilde()

        # see Eq. 10.77 in Bishop
        q_entropy = -0.5 * (loglmbdatilde + D * (np.log(self.kappa_mf/(2*np.pi))-1)) \
            + invwishart_entropy(self.sigma_mf,self.nu_mf)
        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde
            - D*self.kappa_0/self.kappa_mf - self.kappa_0*self.nu_mf*
            np.dot(self.mu_mf -
                self.mu_0,np.linalg.solve(self.sigma_mf,self.mu_mf - self.mu_0))) \
            + invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
            + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self.nu_mf \
            * np.linalg.solve(self.sigma_mf,self.sigma_0).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self, x=None, stats=None):
        assert (x is not None) ^ isinstance(stats, (tuple, np.ndarray))

        if x is not None:
            mu_n, kappa_n, nu_n = self.mu_mf, self.kappa_mf, self.nu_mf
            D = len(mu_n)
            x = np.reshape(x,(-1,D)) - mu_n  # x is now centered
            xs = np.linalg.solve(self.sigma_mf_chol,x.T)

            # see Eqs. 10.64, 10.67, and 10.71 in Bishop
            return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi)
        else:
            D = self.mu_mf.shape[0]

            E_J, E_h, E_muJmuT, E_logdetJ = \
                niw_expectedstats(
                    self.nu_mf, self.sigma_mf, self.mu_mf, self.kappa_mf)

            if isinstance(stats, np.ndarray):
                parammat = np.zeros((D+2,D+2))
                parammat[:D,:D] = E_J
                parammat[:D,-2] = parammat[-2,:D] = -E_h
                parammat[-2,-2] = E_muJmuT
                parammat[-1,-1] = -E_logdetJ

                contract = 'ij,nij->n' if stats.ndim == 3 else 'ij,ij->'
                return -1./2*np.einsum(contract, parammat, stats) \
                    - D/2.*np.log(2*np.pi)
            else:
                x, xxT, n = stats
                c1, c2 = ('i,i->', 'ij,ij->') if x.ndim == 1 \
                    else ('i,ni->n', 'ij,nij->n')

                out = -1./2 * np.einsum(c2, E_J, xxT)
                out += np.einsum(c1, E_h, x)
                out += -n/2.*E_muJmuT
                out += -D/2.*np.log(2*np.pi) + n/2.*E_logdetJ

                return out

    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        D = len(self.mu_0)
        chol = self.sigma_mf_chol
        return special.digamma((self.nu_mf-np.arange(D))/2.).sum() \
            + D*np.log(2) - 2*np.log(chol.diagonal()).sum()

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n, D = getdatasize(data), len(self.mu_0)
        return self._log_partition_function(
            *self._natural_to_standard(
                self.natural_hypparam + self._get_statistics(data,D))) \
            - self._log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
            - n*D/2 * np.log(2*np.pi)

    def _log_partition_function(self,mu,sigma,kappa,nu):
        D = len(mu)
        chol = np.linalg.cholesky(sigma)
        return nu*D/2*np.log(2) + special.multigammaln(nu/2,D) + D/2*np.log(2*np.pi/kappa) \
            - nu*np.log(chol.diagonal()).sum()

    def log_predictive_studentt_datapoints(self,datapoints,olddata):
        D = len(self.mu_0)
        mu_n, sigma_n, kappa_n, nu_n = \
            self._natural_to_standard(
                self.natural_hypparam + self._get_statistics(olddata,D))
        return multivariate_t_loglik(
            datapoints,nu_n-D+1,mu_n,(kappa_n+1)/(kappa_n*(nu_n-D+1))*sigma_n)

    def log_predictive_studentt(self,newdata,olddata):
        newdata = np.atleast_2d(newdata)
        return sum(self.log_predictive_studentt_datapoints(
            d,combinedata((olddata,newdata[:i])))[0] for i,d in enumerate(newdata))

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        D = getdatadimension(data)
        if weights is None:
            statmat = self._get_statistics(data,D)
        else:
            statmat = self._get_weighted_statistics(data,weights,D)

        n, x, xxt = statmat[-1,-1], statmat[-2,:D], statmat[:D,:D]

        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n < D or (np.linalg.svd(xxt,compute_uv=False) > 1e-6).sum() < D:
            self.broken = True
            self.mu = 99999999*np.ones(D)
            self.sigma = np.eye(D)
        else:
            self.mu = x/n
            self.sigma = xxt/n - np.outer(self.mu,self.mu)

        return self

    def MAP(self,data,weights=None):
        D = getdatadimension(data)
        # max likelihood with prior pseudocounts included in data
        if weights is None:
            statmat = self._get_statistics(data)
        else:
            statmat = self._get_weighted_statistics(data,weights)
        statmat += self.natural_hypparam

        n, x, xxt = statmat[-1,-1], statmat[-2,:D], statmat[:D,:D]

        self.mu = x/n
        self.sigma = xxt/n - np.outer(self.mu,self.mu)

        return self


class GaussianFixedMean(_GaussianBase, GibbsSampling, MaxLikelihood):
    def __init__(self,mu=None,sigma=None,nu_0=None,lmbda_0=None):
        self.sigma = sigma

        self.mu = mu

        self.nu_0 = nu_0
        self.lmbda_0 = lmbda_0

        if sigma is None and not any(_ is None for _ in (nu_0,lmbda_0)):
            self.resample()  # initialize from prior

    @property
    def hypparams(self):
        return dict(nu_0=self.nu_0,lmbda_0=self.lmbda_0)

    @property
    def num_parameters(self):
        D = len(self.mu)
        return D*(D+1)/2

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 1e-4:
            if isinstance(data,np.ndarray):
                centered = data[gi(data)] - self.mu
                sumsq = centered.T.dot(centered)
                n = len(centered)
            else:
                sumsq = sum((d[gi(d)]-self.mu).T.dot(d[gi(d)]-self.mu) for d in data)
        else:
            sumsq = None
        return n, sumsq

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > weps:
                centered = data - self.mu
                sumsq = centered.T.dot(weights[:,na]*centered)
            else:
                sumsq = None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > weps:
                sumsq = sum((d-self.mu).T.dot(w[:,na]*(d-self.mu)) for w,d in zip(weights,data))
            else:
                sumsq = None

        return neff, sumsq

    def _posterior_hypparams(self,n,sumsq):
        nu_0, lmbda_0 = self.nu_0, self.lmbda_0
        if n > 1e-4:
            nu_0 = nu_0 + n
            sigma_n = self.lmbda_0 + sumsq
            return sigma_n, nu_0
        else:
            return lmbda_0, nu_0

    ### Gibbs sampling

    def resample(self, data=[]):
        self.sigma = sample_invwishart(*self._posterior_hypparams(
            *self._get_statistics(data)))
        return self

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        D = getdatadimension(data)
        if weights is None:
            n, sumsq = self._get_statistics(data)
        else:
            n, sumsq = self._get_weighted_statistics(data,weights)

        if n < D or (np.linalg.svd(sumsq,compute_uv=False) > 1e-6).sum() < D:
            # broken!
            self.sigma = np.eye(D)*1e-9
            self.broken = True
        else:
            self.sigma = sumsq/n

        return self


class GaussianFixedCov(_GaussianBase, GibbsSampling, MaxLikelihood):
    # See Gelman's Bayesian Data Analysis notation around Eq. 3.18, p. 85
    # in 2nd Edition. We replaced \Lambda_0 with sigma_0 since it is a prior
    # *covariance* matrix rather than a precision matrix. 
    def __init__(self,mu=None,sigma=None,mu_0=None,sigma_0=None):
        self.mu = mu

        self.sigma = sigma

        self.mu_0 = mu_0
        self.sigma_0 = sigma_0

        if mu is None and not any(_ is None for _ in (mu_0,sigma_0)):
            self.resample()

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,sigma_0=self.sigma_0)

    @property
    def sigma_inv(self):
        if not hasattr(self,'_sigma_inv'):
            self._sigma_inv = np.linalg.inv(self.sigma)
        return self._sigma_inv

    @property
    def sigma_inv_0(self):
        if not hasattr(self,'_sigma_inv_0'):
            self._sigma_inv_0 = np.linalg.inv(self.sigma_0)
        return self._sigma_inv_0

    @property
    def num_parameters(self):
        return len(self.mu)

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
            if neff > weps:
                xbar = weights.dot(data) / neff
            else:
                xbar = None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > weps:
                xbar = sum(w.dot(d) for w,d in zip(weights,data)) / neff
            else:
                xbar = None

        return neff, xbar

    def _posterior_hypparams(self,n,xbar):
        # It seems we should be working with lmbda and sigma inv (unless lmbda
        # is a covariance, not a precision)
        sigma_inv, mu_0, sigma_inv_0 = self.sigma_inv, self.mu_0, self.sigma_inv_0
        if n > 0:
            sigma_inv_n = n*sigma_inv + sigma_inv_0
            mu_n = np.linalg.solve(
                sigma_inv_n, sigma_inv_0.dot(mu_0) + n*sigma_inv.dot(xbar))
            return mu_n, sigma_inv_n
        else:
            return mu_0, sigma_inv_0

    ### Gibbs sampling

    def resample(self,data=[]):
        mu_n, sigma_n_inv = self._posterior_hypparams(*self._get_statistics(data))
        D = len(mu_n)
        L = np.linalg.cholesky(sigma_n_inv)
        self.mu = scipy.linalg.solve_triangular(L,np.random.normal(size=D),lower=True) \
            + mu_n
        return self

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, xbar = self._get_statistics(data)
        else:
            n, xbar = self._get_weighted_statistics(data,weights)

        self.mu = xbar
        return self


class GaussianFixed(_FixedParamsMixin, Gaussian):
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

class GaussianNonConj(_GaussianBase, GibbsSampling):
    def __init__(self,mu=None,sigma=None,
            mu_0=None,mu_lmbda_0=None,nu_0=None,sigma_lmbda_0=None):
        self._sigma_distn = GaussianFixedMean(mu=mu,
                nu_0=nu_0,lmbda_0=sigma_lmbda_0,sigma=sigma)
        self._mu_distn = GaussianFixedCov(sigma=self._sigma_distn.sigma,
                mu_0=mu_0, sigma_0=mu_lmbda_0,mu=mu)
        self._sigma_distn.mu = self._mu_distn.mu

    @property
    def hypparams(self):
        d = self._mu_distn.hypparams
        d.update(**self._sigma_distn.hypparams)
        return d

    def _get_mu(self):
        return self._mu_distn.mu

    def _set_mu(self,val):
        self._mu_distn.mu = val
        self._sigma_distn.mu = val

    mu = property(_get_mu,_set_mu)

    def _get_sigma(self):
        return self._sigma_distn.sigma

    def _set_sigma(self,val):
        self._sigma_distn.sigma = val
        self._mu_distn.sigma = val

    sigma = property(_get_sigma,_set_sigma)

    ### Gibbs sampling

    def resample(self,data=[],niter=1):
        if getdatasize(data) == 0:
            niter = 1

        # TODO this is kinda dumb because it collects statistics over and over
        # instead of updating them...
        for itr in range(niter):
            # resample mu
            self._mu_distn.sigma = self._sigma_distn.sigma
            self._mu_distn.resample(data)

            # resample sigma
            self._sigma_distn.mu = self._mu_distn.mu
            self._sigma_distn.resample(data)

        return self


# TODO collapsed
class DiagonalGaussian(_GaussianBase,GibbsSampling,MaxLikelihood,MeanField,Tempering):
    '''
    Product of normal-inverse-gamma priors over mu (mean vector) and sigmas
    (vector of scalar variances).

    The prior follows
        sigmas     ~ InvGamma(alphas_0,betas_0) iid
        mu | sigma ~ N(mu_0,1/nus_0 * diag(sigmas))

    It allows placing different prior hyperparameters on different components.
    '''

    def __init__(self,mu=None,sigmas=None,mu_0=None,nus_0=None,alphas_0=None,betas_0=None):
        # all the s's refer to the fact that these are vectors of length
        # len(mu_0) OR scalars
        if mu_0 is not None:
            D = mu_0.shape[0]
            if nus_0 is not None and \
                    (isinstance(nus_0,int) or isinstance(nus_0,float)):
                nus_0 = nus_0*np.ones(D)
            if alphas_0 is not None and \
                    (isinstance(alphas_0,int) or isinstance(alphas_0,float)):
                alphas_0 = alphas_0*np.ones(D)
            if betas_0 is not None and \
                    (isinstance(betas_0,int) or isinstance(betas_0,float)):
                betas_0 = betas_0*np.ones(D)

        self.mu_0 = self.mf_mu = mu_0
        self.nus_0 = self.mf_nus = nus_0
        self.alphas_0 = self.mf_alphas = alphas_0
        self.betas_0 = self.mf_betas = betas_0

        self.mu = mu
        self.sigmas = sigmas

        assert self.mu is None or (isinstance(self.mu,np.ndarray) and not isinstance(self.mu,np.ma.MaskedArray))
        assert self.sigmas is None or (isinstance(self.sigmas,np.ndarray) and not isinstance(self.sigmas,np.ma.MaskedArray))

        if mu is sigmas is None \
                and not any(_ is None for _ in (mu_0,nus_0,alphas_0,betas_0)):
            self.resample() # intialize from prior

    ### the basics!

    @property
    def parameters(self):
        return self.mu, self.sigmas

    @parameters.setter
    def parameters(self, mu_sigmas_tuple):
        (mu,sigmas) = mu_sigmas_tuple
        self.mu, self.sigmas = mu, sigmas

    @property
    def sigma(self):
        return np.diag(self.sigmas)

    @sigma.setter
    def sigma(self,val):
        val = np.array(val)
        assert val.ndim in (1,2)
        if val.ndim == 1:
            self.sigmas = val
        else:
            self.sigmas = np.diag(val)

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,nus_0=self.nus_0,
                alphas_0=self.alphas_0,betas_0=self.betas_0)

    def rvs(self,size=None):
        size = np.array(size,ndmin=1)
        return np.sqrt(self.sigmas)*\
            np.random.normal(size=np.concatenate((size,self.mu.shape))) + self.mu

    def log_likelihood(self,x,temperature=1.):
        mu, sigmas, D = self.mu, self.sigmas * temperature, self.mu.shape[0]
        x = np.reshape(x,(-1,D))
        Js = -1./(2*sigmas)
        return (np.einsum('ij,ij,j->i',x,x,Js) - np.einsum('ij,j,j->i',x,2*mu,Js)) \
            + (mu**2*Js - 1./2*np.log(2*np.pi*sigmas)).sum()

    ### posterior updating stuff

    @property
    def natural_hypparam(self):
        return self._standard_to_natural(self.alphas_0,self.betas_0,self.mu_0,self.nus_0)

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.alphas_0, self.betas_0, self.mu_0, self.nus_0 = \
            self._natural_to_standard(natparam)

    def _standard_to_natural(self,alphas,betas,mu,nus):
        return np.array([2*betas + nus * mu**2, nus*mu, nus, 2*alphas])

    def _natural_to_standard(self,natparam):
        nus = natparam[2]
        mu = natparam[1] / nus
        alphas = natparam[3]/2.
        betas = (natparam[0] - nus*mu**2) / 2.
        return alphas, betas, mu, nus

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray) and data.shape[0] > 0:
            data = data[gi(data)]
            ns = np.repeat(*data.shape)
            return np.array([
                np.einsum('ni,ni->i',data,data),
                np.einsum('ni->i',data),
                ns,
                ns,
                ])
        else:
            return sum((self._get_statistics(d) for d in data), self._empty_stats())

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            idx = ~np.isnan(data).any(1)
            data = data[idx]
            weights = weights[idx]
            assert data.ndim == 2 and weights.ndim == 1 \
                and data.shape[0] == weights.shape[0]
            neff = np.repeat(weights.sum(),data.shape[1])
            return np.array([weights.dot(data**2), weights.dot(data), neff, neff])
        else:
            return sum(
                (self._get_weighted_statistics(d,w) for d, w in zip(data,weights)),
                self._empty_stats())

    def _empty_stats(self):
        return np.zeros_like(self.natural_hypparam)

    ### Gibbs sampling

    def resample(self,data=[],temperature=1.,stats=None):
        stats = self._get_statistics(data) if stats is None else stats

        alphas_n, betas_n, mu_n, nus_n = self._natural_to_standard(
            self.natural_hypparam + stats / temperature)

        D = mu_n.shape[0]
        self.sigmas = 1/np.random.gamma(alphas_n,scale=1/betas_n)
        self.mu = np.sqrt(self.sigmas/nus_n)*np.random.randn(D) + mu_n

        assert not np.isnan(self.mu).any()
        assert not np.isnan(self.sigmas).any()

        # NOTE: next line is to use Gibbs sampling to initialize mean field
        self.mf_mu = self.mu

        assert self.sigmas.ndim == 1
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigmas = self.sigmas.copy()
        return new

    ### max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data)
        else:
            n, muhat, sumsq = self._get_weighted_statistics_old(data,weights)

        self.mu = muhat
        self.sigmas = sumsq/n

        return self

    ### Mean Field

    @property
    def mf_natural_hypparam(self):
        return self._standard_to_natural(self.mf_alphas,self.mf_betas,self.mf_mu,self.mf_nus)

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.mf_alphas, self.mf_betas, self.mf_mu, self.mf_nus = \
            self._natural_to_standard(natparam)
        # NOTE: this part is for plotting
        self.mu = self.mf_mu
        self.sigmas = np.where(self.mf_alphas > 1,self.mf_betas / (self.mf_alphas - 1),100000)

    def meanfieldupdate(self,data,weights):
        self.mf_natural_hypparam = \
            self.natural_hypparam + self._get_weighted_statistics(data,weights)

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                self.natural_hypparam
                + 1./prob * self._get_weighted_statistics(data,weights))

    def get_vlb(self):
        natparam_diff = self.natural_hypparam - self.mf_natural_hypparam
        expected_stats = self._expected_statistics(
            self.mf_alphas,self.mf_betas,self.mf_mu,self.mf_nus)
        linear_term = sum(v1.dot(v2) for v1, v2 in zip(natparam_diff, expected_stats))

        normalizer_term = \
            self._log_Z(self.alphas_0,self.betas_0,self.mu_0,self.nus_0) \
            - self._log_Z(self.mf_alphas,self.mf_betas,self.mf_mu,self.mf_nus)

        return linear_term - normalizer_term - len(self.mf_mu)/2. * np.log(2*np.pi)

    def expected_log_likelihood(self,x):
        x = np.atleast_2d(x).reshape((-1,len(self.mf_mu)))
        a,b,c,d = self._expected_statistics(
            self.mf_alphas,self.mf_betas,self.mf_mu,self.mf_nus)
        return (x**2).dot(a) + x.dot(b) + c.sum() + d.sum() \
            - len(self.mf_mu)/2. * np.log(2*np.pi)

    def _expected_statistics(self,alphas,betas,mu,nus):
        return np.array([
            -1./2 * alphas/betas,
            mu * alphas/betas,
            -1./2 * (1./nus + mu**2 * alphas/betas),
            -1./2 * (np.log(betas) - special.digamma(alphas))])

    def _log_Z(self,alphas,betas,mu,nus):
        return (special.gammaln(alphas) - alphas*np.log(betas) - 1./2*np.log(nus)).sum()

# TODO meanfield
class DiagonalGaussianNonconjNIG(_GaussianBase,GibbsSampling):
    '''
    Product of normal priors over mu and product of gamma priors over sigmas.
    Note that while the conjugate prior in DiagonalGaussian is of the form
    p(mu,sigmas), this prior is of the form p(mu)p(sigmas). Therefore its
    resample() update has to perform inner iterations.

    The prior follows
        mu     ~ N(mu_0,diag(sigmas_0))
        sigmas ~ InvGamma(alpha_0,beta_0) iid
    '''

    def __init__(self,mu=None,sigmas=None,mu_0=None,sigmas_0=None,alpha_0=None,beta_0=None,
            niter=20):
        self.mu_0, self.sigmas_0 = mu_0, sigmas_0
        self.alpha_0, self.beta_0 = alpha_0, beta_0

        self.niter = niter

        if None in (mu,sigmas):
            self.resample()
        else:
            self.mu, self.sigmas = mu, sigmas

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,sigmas_0=self.sigmas_0,alpha_0=self.alpha_0,beta_0=self.beta_0)

    # TODO next three methods are copied from DiagonalGaussian, factor them out

    @property
    def sigma(self):
        return np.diag(self.sigmas)

    def rvs(self,size=None):
        size = np.array(size,ndmin=1)
        return np.sqrt(self.sigmas)*\
            np.random.normal(size=np.concatenate((size,self.mu.shape))) + self.mu

    def log_likelihood(self,x):
        mu, sigmas, D = self.mu, self.sigmas, self.mu.shape[0]
        x = np.reshape(x,(-1,D))
        Js = -1./(2*sigmas)
        return (np.einsum('ij,ij,j->i',x,x,Js) - np.einsum('ij,j,j->i',x,2*mu,Js)) \
            + (mu**2*Js - 1./2*np.log(2*np.pi*sigmas)).sum()


    def resample(self,data=[]):
        n, y, ysq = self._get_statistics(data)
        if n == 0:
            self.mu = np.sqrt(self.sigmas_0) * np.random.randn(self.mu_0.shape[0]) + self.mu_0
            self.sigmas = 1./np.random.gamma(self.alpha_0,scale=1./self.beta_0)
        else:
            for itr in range(self.niter):
                sigmas_n = 1./(1./self.sigmas_0 + n / self.sigmas)
                mu_n = (self.mu_0 / self.sigmas_0 + y / self.sigmas) * sigmas_n
                self.mu = np.sqrt(sigmas_n) * np.random.randn(mu_n.shape[0]) + mu_n

                alphas_n = self.alpha_0 + 1./2*n
                betas_n = self.beta_0 + 1./2*(ysq + n*self.mu**2 - 2*self.mu*y)
                self.sigmas = 1./np.random.gamma(alphas_n,scale=1./betas_n)
        return self

    def _get_statistics(self,data):
        # TODO dont forget to handle nans
        assert isinstance(data,(list,np.ndarray)) and not isinstance(data,np.ma.MaskedArray)
        if isinstance(data,np.ndarray):
            data = data[gi(data)]
            n = data.shape[0]
            y = np.einsum('ni->i',data)
            ysq = np.einsum('ni,ni->i',data,data)
            return np.array([n,y,ysq],dtype=np.object)
        else:
            return sum((self._get_statistics(d) for d in data),self._empty_stats)

    @property
    def _empty_stats(self):
        return np.array([0.,np.zeros_like(self.mu_0),np.zeros_like(self.mu_0)],
                dtype=np.object)

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

    def __init__(self,mu=None,sigma=None,mu_0=None,nu_0=None,alpha_0=None,beta_0=None):
        self.mu = mu
        self.sigma = sigma

        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if mu is sigma is None and not any(_ is None for _ in (mu_0,nu_0,alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,nu_0=self.nu_0,alpha_0=self.alpha_0,beta_0=self.beta_0)

    def rvs(self,size=None):
        return np.sqrt(self.sigma)*np.random.normal(size=tuple(size)+self.mu.shape) + self.mu

    def log_likelihood(self,x):
        mu, sigma, D = self.mu, self.sigma, self.mu.shape[0]
        x = np.reshape(x,(-1,D))
        return (-0.5*((x-mu)**2).sum(1)/sigma - D*np.log(np.sqrt(2*np.pi*sigma)))

    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, nu_0, alpha_0, beta_0 = self.mu_0, self.nu_0, self.alpha_0, self.beta_0
        D = mu_0.shape[0]
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
        mu_n, nu_n, alpha_n, beta_n = self._posterior_hypparams(
            *self._get_statistics(data, D=self.mu_0.shape[0]))
        D = mu_n.shape[0]
        self.sigma = 1/np.random.gamma(alpha_n,scale=1/beta_n)
        self.mu = np.sqrt(self.sigma/nu_n)*np.random.randn(D)+mu_n
        return self

    def _get_statistics(self,data, D=None):
        n = getdatasize(data)
        if n > 0:
            D = D if D else getdatadimension(data)
            if isinstance(data,np.ndarray):
                assert (data.ndim == 1 and data.shape == (D,)) \
                    or (data.ndim == 2 and data.shape[1] == D)
                data = np.reshape(data,(-1,D))
                xbar = data.mean(0)
                sumsq = ((data-xbar)**2).sum()
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(((np.reshape(data,(-1,D)) - xbar)**2).sum() for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq


class _ScalarGaussianBase(object):
    @property
    def params(self):
        return dict(mu=self.mu,sigmasq=self.sigmasq)

    def rvs(self,size=None):
        return np.sqrt(self.sigmasq)*np.random.normal(size=size)+self.mu

    def log_likelihood(self,x):
        x = np.reshape(x,(-1,1))
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).ravel()

    def __repr__(self):
        return self.__class__.__name__ + '(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)

    def plot(self,data=None,indices=None,color='b',plot_params=True,label=None):
        import matplotlib.pyplot as plt
        data = np.concatenate(data) if data is not None else None
        indices = np.concatenate(indices) if indices is not None else None

        if data is not None:
            assert indices is not None
            plt.plot(indices,data,color=color,marker='x',linestyle='')

        if plot_params:
            assert indices is not None
            if len(indices) > 1:
                from util.general import rle
                vals, lens = rle(np.diff(indices))
                starts = np.concatenate(((0,),lens.cumsum()[:-1]))
                for start, blocklen in zip(starts[vals == 1], lens[vals == 1]):
                    plt.plot(indices[start:start+blocklen],
                            np.repeat(self.mu,blocklen),color=color,linestyle='--')
            else:
                plt.plot(indices,[self.mu],color=color,marker='+')

    ### mostly shared statistics gathering

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                ybar = data.mean()
                centered = data.ravel() - ybar
                sumsqc = centered.dot(centered)
            elif isinstance(data,list):
                ybar = sum(d.sum() for d in data)/n
                sumsqc = sum((d.ravel()-ybar).dot(d.ravel()-ybar) for d in data)
            else:
                ybar = data
                sumsqc = 0
        else:
            ybar = None
            sumsqc = None

        return n, ybar, sumsqc

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > weps:
                ybar = weights.dot(data.ravel()) / neff
                centered = data.ravel() - ybar
                sumsqc = centered.dot(weights*centered)
            else:
                ybar = None
                sumsqc = None
        elif isinstance(data,list):
            neff = sum(w.sum() for w in weights)
            if neff > weps:
                ybar = sum(w.dot(d.ravel()) for d,w in zip(data,weights)) / neff
                sumsqc = sum((d.ravel()-ybar).dot(w*(d.ravel()-ybar))
                        for d,w in zip(data,weights))
            else:
                ybar = None
                sumsqc = None
        else:
            ybar = data
            sumsqc = 0

        return neff, ybar, sumsqc

    ### max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, ybar, sumsqc = self._get_statistics(data)
        else:
            n, ybar, sumsqc = self._get_weighted_statistics(data,weights)

        if sumsqc > 0:
            self.mu = ybar
            self.sigmasq = sumsqc/n
        else:
            self.broken = True
            self.mu = 999999999.
            self.sigmsq = 1.

        return self

class ScalarGaussianNIX(_ScalarGaussianBase, GibbsSampling, Collapsed):
    '''
    Conjugate Normal-(Scaled-)Inverse-ChiSquared prior. (Another parameterization is the
    Normal-Inverse-Gamma.)
    '''
    def __init__(self,mu=None,sigmasq=None,mu_0=None,kappa_0=None,sigmasq_0=None,nu_0=None):
        self.mu = mu
        self.sigmasq = sigmasq

        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.sigmasq_0 = sigmasq_0
        self.nu_0 = nu_0

        if mu is sigmasq is None \
                and not any(_ is None for _ in (mu_0,kappa_0,sigmasq_0,nu_0)):
            self.resample() # intialize from prior

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,kappa_0=self.kappa_0,
                sigmasq_0=self.sigmasq_0,nu_0=self.nu_0)

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
        return self

    ### Collapsed

    def log_marginal_likelihood(self,data):
        n = getdatasize(data)
        kappa_0, sigmasq_0, nu_0 = self.kappa_0, self.sigmasq_0, self.nu_0
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        return special.gammaln(nu_n/2) - special.gammaln(nu_0/2) \
            + 0.5*(np.log(kappa_0) - np.log(kappa_n)
                   + nu_0 * (np.log(nu_0) + np.log(sigmasq_0))
                   - nu_n * (np.log(nu_n) + np.log(sigmasq_n))
                   - n*np.log(np.pi))

    def log_predictive_single(self,y,olddata):
        # mostly for testing or speed
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata))
        return stats.t.logpdf(y,nu_n,loc=mu_n,scale=np.sqrt((1+kappa_n)*sigmasq_n/kappa_n))


class ScalarGaussianNonconjNIX(_ScalarGaussianBase, GibbsSampling):
    '''
    Non-conjugate separate priors on mean and variance parameters, via
    mu ~ Normal(mu_0,tausq_0)
    sigmasq ~ (Scaled-)Inverse-ChiSquared(sigmasq_0,nu_0)
    '''
    def __init__(self,mu=None,sigmasq=None,mu_0=None,tausq_0=None,sigmasq_0=None,nu_0=None,
            niter=1):
        self.mu, self.sigmasq = mu, sigmasq
        self.mu_0, self.tausq_0 = mu_0, tausq_0
        self.sigmasq_0, self.nu_0 = sigmasq_0, nu_0

        self.niter = niter

        if mu is sigmasq is None \
                and not any(_ is None for _ in (mu_0, tausq_0, sigmasq_0, nu_0)):
            self.resample() # intialize from prior

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,tausq_0=self.tausq_0,
                sigmasq_0=self.sigmasq_0,nu_0=self.nu_0)

    def resample(self,data=[],niter=None):
        n = getdatasize(data)
        niter = self.niter if niter is None else niter
        if n > 0:
            data = flattendata(data)
            datasum = data[gi(data)].sum()
            datasqsum = (data[gi(data)]**2).sum()
            nu_n = self.nu_0 + n
            for itr in range(niter):
                # resample mean
                tausq_n = 1/(1/self.tausq_0 + n/self.sigmasq)
                mu_n = tausq_n*(self.mu_0/self.tausq_0 + datasum/self.sigmasq)
                self.mu = np.sqrt(tausq_n)*np.random.normal() + mu_n
                # resample variance
                sigmasq_n = (self.nu_0*self.sigmasq_0 + (datasqsum + n*self.mu**2-2*datasum*self.mu))/nu_n
                self.sigmasq = sigmasq_n*nu_n/np.random.chisquare(nu_n)
        else:
            self.mu = np.sqrt(self.tausq_0) * np.random.normal() + self.mu_0
            self.sigmasq = self.sigmasq_0*self.nu_0/np.random.chisquare(self.nu_0)

        return self

class ScalarGaussianNonconjNIG(_ScalarGaussianBase, MeanField, MeanFieldSVI):
    # NOTE: this is like ScalarGaussianNonconjNiIG except prior is in natural
    # coordinates

    def __init__(self,h_0,J_0,alpha_0,beta_0,
            mu=None,sigmasq=None,
            h_mf=None,J_mf=None,alpha_mf=None,beta_mf=None,niter=1):
        self.h_0, self.J_0 = h_0, J_0
        self.alpha_0, self.beta_0 = alpha_0, beta_0

        self.h_mf = h_mf if h_mf is not None else J_0 * np.random.normal(h_0/J_0,1./np.sqrt(J_0))
        self.J_mf = J_mf if J_mf is not None else J_0
        self.alpha_mf = alpha_mf if alpha_mf is not None else alpha_0
        self.beta_mf = beta_mf if beta_mf is not None else beta_0

        self.niter = niter

        self.mu = mu if mu is not None else np.random.normal(h_0/J_0,1./np.sqrt(J_0))
        self.sigmasq = sigmasq if sigmasq is not None else 1./np.random.gamma(alpha_0,1./beta_0)

    @property
    def hypparams(self):
        return dict(h_0=self.h_0,J_0=self.J_0,alpha_0=self.alpha_0,beta_0=self.beta_0)

    @property
    def _E_mu(self):
        # E[mu], E[mu**2]
        return self.h_mf / self.J_mf, 1./self.J_mf + (self.h_mf / self.J_mf)**2

    @property
    def _E_sigmasq(self):
        # E[1/sigmasq], E[ln sigmasq]
        return self.alpha_mf / self.beta_mf, \
            np.log(self.beta_mf) - special.digamma(self.alpha_mf)

    @property
    def natural_hypparam(self):
        return np.array([self.alpha_0,self.beta_0,self.h_0,self.J_0])

    @natural_hypparam.setter
    def natural_hypparam(self,natural_hypparam):
        self.alpha_0, self.beta_0, self.h_0, self.J_0 = natural_hypparam

    @property
    def mf_natural_hypparam(self):
        return np.array([self.alpha_mf,self.beta_mf,self.h_mf,self.J_mf])

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,mf_natural_hypparam):
        self.alpha_mf, self.beta_mf, self.h_mf, self.J_mf = mf_natural_hypparam
        # set point estimates of (mu, sigmasq) for plotting and stuff
        self.mu, self.sigmasq = self.h_mf / self.J_mf, self.beta_mf / (self.alpha_mf-1)

    def _resample_from_mf(self):
        self.mu, self.sigmasq = np.random.normal(self.h_mf/self.J_mf,np.sqrt(1./self.J_mf)), \
            np.random.gamma(self.alpha_mf,1./self.beta_mf)
        return self

    def expected_log_likelihood(self,x):
        (Emu, Emu2), (Esigmasqinv, Elnsigmasq) = self._E_mu, self._E_sigmasq
        return -1./2 * Esigmasqinv * (x**2 + Emu2 - 2*x*Emu) \
            - 1./2*Elnsigmasq - 1./2*np.log(2*np.pi)

    def get_vlb(self):
        # E[ln p(mu) / q(mu)] part
        h_0, J_0, J_mf = self.h_0, self.J_0, self.J_mf
        Emu, Emu2 = self._E_mu
        p_mu_avgengy = -1./2*J_0*Emu2 + h_0*Emu \
            - 1./2*(h_0**2/J_0) + 1./2*np.log(J_0) - 1./2*np.log(2*np.pi)
        q_mu_entropy = 1./2*np.log(2*np.pi*np.e/J_mf)

        # E[ln p(sigmasq) / q(sigmasq)] part
        alpha_0, beta_0, alpha_mf, beta_mf = \
            self.alpha_0, self.beta_0, self.alpha_mf, self.beta_mf
        (Esigmasqinv, Elnsigmasq) = self._E_sigmasq
        p_sigmasq_avgengy = (-alpha_0-1)*Elnsigmasq + (-beta_0)*Esigmasqinv \
            - (special.gammaln(alpha_0) - alpha_0*np.log(beta_0))
        q_sigmasq_entropy = alpha_mf + np.log(beta_mf) + special.gammaln(alpha_mf) \
            - (1+alpha_mf)*special.digamma(alpha_mf)

        return p_mu_avgengy + q_mu_entropy + p_sigmasq_avgengy + q_sigmasq_entropy

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        # like meanfieldupdate except we step the factors simultaneously

        # NOTE: unlike the fully conjugate case, there are interaction terms, so
        # we work on the destructured pieces
        neff, y, ysq = self._get_weighted_statistics(data,weights)
        Emu, _ = self._E_mu
        Esigmasqinv, _ = self._E_sigmasq


        # form new natural hyperparameters as if doing a batch update
        alpha_new = self.alpha_0 + 1./prob * 1./2*neff
        beta_new = self.beta_0 + 1./prob * 1./2*(ysq + neff*Emu**2 - 2*Emu*y)

        h_new = self.h_0 + 1./prob * Esigmasqinv * y
        J_new = self.J_0 + 1./prob * Esigmasqinv * neff


        # take a step
        self.alpha_mf = (1-stepsize)*self.alpha_mf + stepsize*alpha_new
        self.beta_mf = (1-stepsize)*self.beta_mf + stepsize*beta_new

        self.h_mf = (1-stepsize)*self.h_mf + stepsize*h_new
        self.J_mf = (1-stepsize)*self.J_mf + stepsize*J_new

        # calling this setter will set point estimates for (mu,sigmasq) for
        # plotting and sampling and stuff
        self.mf_natural_hypparam = (self.alpha_mf, self.beta_mf, self.h_mf, self.J_mf)

        return self

    def meanfieldupdate(self,data,weights,niter=None):
        niter = niter if niter is not None else self.niter
        neff, y, ysq = self._get_weighted_statistics(data,weights)
        for niter in range(niter):
            # update q(sigmasq)
            Emu, _ = self._E_mu

            self.alpha_mf = self.alpha_0 + 1./2*neff
            self.beta_mf = self.beta_0 + 1./2*(ysq + neff*Emu**2 - 2*Emu*y)

            # update q(mu)
            Esigmasqinv, _ = self._E_sigmasq

            self.h_mf = self.h_0 + Esigmasqinv * y
            self.J_mf = self.J_0 + Esigmasqinv * neff

        # calling this setter will set point estimates for (mu,sigmasq) for
        # plotting and sampling and stuff
        self.mf_natural_hypparam = \
            (self.alpha_mf, self.beta_mf, self.h_mf, self.J_mf)

        return self

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            y = weights.dot(data)
            ysq = weights.dot(data**2)
        else:
            return sum(
                self._get_weighted_statistics(d,w) for d,w in zip(data,weights))
        return np.array([neff,y,ysq])


class ScalarGaussianFixedvar(_ScalarGaussianBase, GibbsSampling):
    '''
    Conjugate normal prior on mean.
    '''
    def __init__(self,mu=None,sigmasq=None,mu_0=None,tausq_0=None):
        self.mu = mu

        self.sigmasq = sigmasq

        self.mu_0 = mu_0
        self.tausq_0 = tausq_0

        if mu is None and not any(_ is None for _ in (mu_0,tausq_0)):
            self.resample()  # intialize from prior

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,tausq_0=self.tausq_0)

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
        return self

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                xbar = data.mean()
            else:
                xbar = sum(d.sum() for d in data)/n
        else:
            xbar = None
        return n, xbar

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
        else:
            neff = sum(w.sum() for w in weights)

        if neff > weps:
            if isinstance(data,np.ndarray):
                xbar = data.dot(weights) / neff
            else:
                xbar = sum(w.dot(d) for d,w in zip(data,weights)) / neff
        else:
            xbar = None

        return neff, xbar

    def max_likelihood(self,data,weights=None):
        if weights is None:
            _, xbar = self._get_statistics(data)
        else:
            _, xbar = self._get_weighted_statistics(data,weights)

        self.mu = xbar
