from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
from numpy.core.umath_tests import inner1d
import scipy.linalg
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import abc
import copy
from warnings import warn

import pybasicbayes
from pybasicbayes.abstractions import Distribution, BayesianDistribution, \
        GibbsSampling, MeanField, MeanFieldSVI, Collapsed, MaxLikelihood, MAP, Tempering
from pybasicbayes.util.stats import sample_niw, sample_mniw, sample_invwishart, invwishart_entropy,\
        invwishart_log_partitionfunction, sample_discrete, sample_pareto,\
        sample_discrete_from_log, getdatasize, flattendata,\
        getdatadimension, combinedata, multivariate_t_loglik, gi, atleast_2d
from pybasicbayes.util.general import blockarray, inv_psd, solve_psd, cumsum
try:
    from pybasicbayes.util.cstats import sample_crp_tablecounts
except ImportError:
    from pybasicbayes.util.stats import sample_crp_tablecounts

# Threshold on weights to perform posterior computation
weps = 1e-12

# TODO reduce reallocation of parameters

##########
#  Meta  #
##########

class _FixedParamsMixin(Distribution):
    @property
    def num_parameters(self):
        return 0

    def resample(self,*args,**kwargs):
        return self

    def meanfieldupdate(self,*args,**kwargs):
        return self

    def get_vlb(self):
        return 0.

    def copy_sample(self):
        return self

class ProductDistribution(GibbsSampling,MaxLikelihood):
    # TODO make a better __repr__

    def __init__(self,distns,slices=None):
        self._distns = distns
        self._slices = slices if slices is not None else \
                [slice(i,i+1) for i in xrange(len(distns))]

    @property
    def params(self):
        return {idx:distn.params for idx,distn in enumerate(self._distns)}

    @property
    def hypparams(self):
        return {idx:distn.hypparams for idx,distn in enumerate(self._distns)}

    @property
    def num_parameters(self):
        return sum(d.num_parameters for d in self._distns)

    @staticmethod
    def atleast_2d(data):
        # NOTE: can't use np.atleast_2d because if it's 1D we want axis 1 to be
        # the singleton
        if data.ndim == 1:
            return data.reshape((-1,1))
        return data

    def rvs(self,size=[]):
        return np.concatenate([self.atleast_2d(distn.rvs(size=size))
            for distn in self._distns],axis=-1)

    def log_likelihood(self,x):
        return sum(distn.log_likelihood(x[...,sl])
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
        return np.sum([distn.expected_log_likelihood(x[...,sl])
                for distn,sl in zip(self._distns,self._slices)], axis=0).ravel()

    def meanfieldupdate(self,data,weights,**kwargs):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfieldupdate(data[...,sl],weights)
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfieldupdate([d[...,sl] for d in data],weights=weights)
        return self

    def _resample_from_mf(self):
        for distn in self._distns:
            distn._resample_from_mf()

    ### SVI

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfield_sgdstep(data[...,sl],weights,minibatchfrac,stepsize)
        else:
            for distn,sl in zip(self._distns,self._slices):
                distn.meanfield_sgdstep([d[...,sl] for d in data],weights,minibatchfrac,stepsize)
        return self

################
#  Continuous  #
################

class Regression(GibbsSampling):
    def __init__(self,
            nu_0=None,S_0=None,M_0=None,K_0=None,
            affine=False,
            A=None,sigma=None):
        self.affine = affine

        self.A = A
        self.sigma = sigma

        if not any(_ is None for _ in (nu_0,S_0,M_0,K_0)):
            self.natural_hypparam = self._standard_to_natural(nu_0,S_0,M_0,K_0)

        if A is sigma is None and not any(_ is None for _ in (nu_0,S_0,M_0,K_0)):
            self.resample() # initialize from prior

    @property
    def parameters(self):
        return (self.A, self.sigma)

    @parameters.setter
    def parameters(self,(A,sigma)):
        self.A = A
        self.sigma = sigma

    @property
    def D_in(self):
        # NOTE: D_in includes the extra affine coordinate
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[1]

    @property
    def D_out(self):
        mat = self.A if self.A is not None else self.natural_hypparam[1]
        return mat.shape[0]

    ### converting between natural and standard parameters

    @staticmethod
    def _standard_to_natural(nu,S,M,K):
        Kinv = inv_psd(K)
        A = S + M.dot(Kinv).dot(M.T)
        B = M.dot(Kinv)
        C = Kinv
        d = nu
        return np.array([A,B,C,d])

    @staticmethod
    def _natural_to_standard(natparam):
        A,B,C,d = natparam
        nu = d
        Kinv = C
        K = inv_psd(Kinv)
        M = B.dot(K)
        S = A - B.dot(K).dot(B.T)

        # numerical padding here...
        K += 1e-8*np.eye(K.shape[0])
        assert np.all(0 < np.linalg.eigvalsh(S))
        assert np.all(0 < np.linalg.eigvalsh(K))

        return nu, S, M, K

    ### getting statistics

    def _get_statistics(self,data):
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),self._empty_statistics())
        else:
            data = data[~np.isnan(data).any(1)]
            n, D = data.shape[0], self.D_out

            statmat = data.T.dot(data)
            xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = data.sum(0)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,list):
            return sum((self._get_statistics(d) for d in data),self._empty_statistics())
        else:
            gi = ~np.isnan(data).any(1)
            data, weights = data[gi], weights[gi]
            neff, D = weights.sum(), self.D_out

            statmat = data.T.dot(weights[:,na]*data)
            xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]

            if self.affine:
                xy = weights.dot(data)
                x, y = xy[:-D], xy[-D:]
                xxT = blockarray([[xxT,x[:,na]],[x[na,:],np.atleast_2d(n)]])
                yxT = np.hstack((yxT,y[:,na]))

            return np.array([yyT, yxT, xxT, n])


    def _empty_statistics(self):
        D_in, D_out = self.D_in, self.D_out
        return np.array(
            [np.zeros((D_out,D_out)),np.zeros((D_out,D_in)),np.zeros((D_in,D_in)),0])

    ### distribution

    def log_likelihood(self,xy):
        A, sigma, D = self.A, self.sigma, self.D_out
        x, y = xy[:,:-D], xy[:,-D:]

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        sigma_inv = np.linalg.inv(sigma)
        parammat = -1./2 * blockarray([
            [A.T.dot(sigma_inv).dot(A), -A.T.dot(sigma_inv)],
            [-sigma_inv.dot(A), sigma_inv]
            ])
        out = np.einsum('ni,ni->n',xy.dot(parammat),xy)
        out -= D/2*np.log(2*np.pi) + np.log(np.diag(np.linalg.cholesky(sigma))).sum()

        if self.affine:
            out += y.dot(sigma_inv).dot(b)
            out -= x.dot(A.T).dot(sigma_inv).dot(b)
            out -= 1./2*b.dot(sigma_inv).dot(b)

        return out

    def rvs(self,x=None,size=1,return_xy=True):
        A, sigma = self.A, self.sigma

        if self.affine:
            A, b = A[:,:-1], A[:,-1]

        x = np.random.normal(size=(size,A.shape[1])) if x is None else x
        y = x.dot(A.T) + np.random.normal(size=(x.shape[0],self.D_out))\
                .dot(np.linalg.cholesky(sigma).T)

        if self.affine:
            y += b.T

        return np.hstack((x,y)) if return_xy else y

    ### Gibbs sampling

    def resample(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        self.A, self.sigma = sample_mniw(
                *self._natural_to_standard(self.natural_hypparam + stats))

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            stats = self._get_statistics(data)
        else:
            stats = self._get_weighted_statistics(data)

        yyT, yxT, xxT, n = stats

        if n > 0:
            try:
                self.A = np.linalg.solve(xxT, yxT.T).T
                self.sigma = (yyT - self.A.dot(yxT.T))/n
            except np.linalg.LinAlgError:
                self.broken = True
        else:
            self.broken = True

        return self

    def empirical_bayes(self,data,D_in,D_out):
        self.A = np.zeros((D_out,D_in)) # so self.D_in, self.D_out work
        self.natural_hypparam = self._get_statistics(data)
        self.resample() # intialize from prior given new hyperparameters
        return self


class ARDRegression(Regression):
    def __init__(self,
            a,b,nu_0,S_0,M_0,
            blocksizes=None,K_0=None,niter=10,**kwargs):
        blocksizes = np.ones(M_0.shape[1],dtype=np.int64) if blocksizes is None else blocksizes
        self.niter = niter
        self.blocksizes = np.array(blocksizes)
        self.starts = cumsum(blocksizes,strict=True)
        self.stops = cumsum(blocksizes,strict=False)

        self.a = np.repeat(a,len(blocksizes))
        self.b = np.repeat(b,len(blocksizes))

        self.nu_0 = nu_0
        self.S_0 = S_0
        self.M_0 = M_0

        if K_0 is None:
            self.resample_K()
        else:
            self.K_0 = K_0

        super(ARDRegression,self).__init__(K_0=self.K_0,nu_0=nu_0,S_0=S_0,M_0=M_0,**kwargs)

    def resample(self,data=[],stats=None):
        if len(data) > 0 or stats is not None:
            stats = self._get_statistics(data) if stats is None else stats
            for itr in xrange(self.niter):
                self.A, self.sigma = sample_mniw(
                        *self._natural_to_standard(self.natural_hypparam + stats))

                mat = self.M_0 - self.A
                self.resample_K(1./2*np.einsum(
                    'ij,ij->j',mat,np.linalg.solve(self.sigma,mat)))
        else:
            self.resample_K()
            super(ARDRegression,self).resample()

    def resample_K(self,diag=None):
        if diag is None:
            a, b = self.a, self.b
        else:
            sums = [diag[start:stop].sum() for start,stop in zip(self.starts,self.stops)]
            a = self.a + self.D_out*self.blocksizes/2.
            b = self.b + np.array(sums)

        ks = 1./np.random.gamma(a,scale=1./b)
        self.K_0 = np.diag(np.repeat(ks,self.blocksizes))

        self.natural_hypparam = self._standard_to_natural(self.nu_0,self.S_0,self.M_0,self.K_0)

class _GaussianBase(object):
    @property
    def params(self):
        return dict(mu=self.mu,sigma=self.sigma)

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
        size = size + (self.mu.shape[0],) if isinstance(size,tuple) else (size,self.mu.shape[0])
        return self.mu + np.random.normal(size=size).dot(self.sigma_chol.T)

    def log_likelihood(self,x):
        try:
            mu, sigma, D = self.mu, self.sigma, self.D
            sigma_chol = self.sigma_chol
            bads = np.isnan(np.atleast_2d(x)).any(axis=1)
            x = np.nan_to_num(x).reshape((-1,D)) - mu
            xs = scipy.linalg.solve_triangular(sigma_chol,x.T,lower=True)
            out = -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) \
                    - np.log(sigma_chol.diagonal()).sum()
            out[bads] = 0
            return out
        except np.linalg.LinAlgError:
            # NOTE: degenerate distribution doesn't have a density on the full
            # space. I need a mechanism for getting the density on a subspace...
            return np.repeat(-np.inf,x.shape[0])

    ### plotting

    # TODO making animations, this seems to generate an extra notebook figure

    _scatterplot = None
    _parameterplot = None

    def plot(self,ax=None,data=None,indices=None,color='b',plot_params=True,label='',alpha=1.,
            update=False,draw=True):
        from util.plot import project_data, plot_gaussian_projection, plot_gaussian_2D
        ax = ax if ax else plt.gca()
        D = self.D
        if data is not None:
            data = flattendata(data)

        if data is not None:
            if D > 2:
                data = project_data(data,self.plotting_subspace_basis)
            if update and self._scatterplot is not None:
                self._scatterplot.set_offsets(data)
                self._scatterplot.set_color(color)
            else:
                self._scatterplot = ax.scatter(data[:,0],data[:,1],marker='.',color=color)

        if plot_params:
            if D > 2:
                self._parameterplot = \
                    plot_gaussian_projection(self.mu,self.sigma,self.plotting_subspace_basis,
                            color=color,label=label,alpha=min(1-1e-3,alpha),
                            ax=ax, artists=self._parameterplot if update else None)
            else:
                self._parameterplot = \
                    plot_gaussian_2D(self.mu,self.sigma,color=color,label=label,alpha=min(1-1e-3,alpha),
                            ax=ax, artists=self._parameterplot if update else None)

        if draw: plt.draw()

        return [self._scatterplot] + list(self._parameterplot)

    def to_json_dict(self):
        D = self.mu.shape[0]
        assert D == 2
        U,s,_ = np.linalg.svd(self.sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),'ry':np.sqrt(s[1]),
                'theta':theta}


class Gaussian(_GaussianBase, GibbsSampling, MeanField, MeanFieldSVI, Collapsed, MAP, MaxLikelihood):
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

    def __init__(self,mu=None,sigma=None,mu_0=None,sigma_0=None,kappa_0=None,nu_0=None):
        self.mu    = mu
        self.sigma = sigma

        # NOTE: resampling will set mu_mf and sigma_mf
        self.mu_0    = self.mu_mf    = mu_0
        self.sigma_0 = self.sigma_mf = sigma_0
        self.kappa_0 = self.kappa_mf = kappa_0
        self.nu_0    = self.nu_mf    = nu_0

        if mu is sigma is None \
                and not any(_ is None for _ in (mu_0,sigma_0,kappa_0,nu_0)):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,sigma_0=self.sigma_0,kappa_0=self.kappa_0,nu_0=self.nu_0)

    @property
    def natural_hypparam(self):
        return self._standard_to_natural(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0)

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.mu_0, self.sigma_0, self.kappa_0, self.nu_0 = self._natural_to_standard(natparam)

    def _standard_to_natural(self,mu_mf,sigma_mf,kappa_mf,nu_mf):
        D = sigma_mf.shape[0]
        out = np.zeros((D+2,D+2))
        out[:D,:D] = sigma_mf + kappa_mf * np.outer(mu_mf,mu_mf)
        out[:D,-2] = out[-2,:D] = kappa_mf * mu_mf
        out[-2,-2] = kappa_mf
        out[-1,-1] = nu_mf
        return out

    def _natural_to_standard(self,natparam):
        D = natparam.shape[0]-2
        A = natparam[:D,:D]
        b = natparam[:D,-2]
        c = natparam[-2,-2]
        d = natparam[-1,-1]
        return b/c, A - np.outer(b,b)/c, c, d

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
        else:
            return None

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
            return sum(map(self._get_statistics,data),out)

    def _get_weighted_statistics(self,data,weights,D=None):
        D = getdatadimension(data) if D is None else D
        out = np.zeros((D+2,D+2))
        if isinstance(data,np.ndarray):
            out[:D,:D] = data.T.dot(weights[:,na]*data)
            out[-2,:D] = out[:D,-2] = weights.dot(data)
            out[-2,-2] = out[-1,-1] = weights.sum()
            return out
        else:
            return sum(map(self._get_weighted_statistics,data,weights),out)

    def _get_empty_statistics(self, D):
        out = np.zeros((D+2,D+2))
        return out

    def empirical_bayes(self,data):
        self.natural_hypparam = self._get_statistics(data)
        self.resample() # intialize from prior given new hyperparameters
        return self

    ### Gibbs sampling

    def resample(self,data=[]):
        D = len(self.mu_0)
        self.mu, self.sigma = \
                sample_niw(*self._natural_to_standard(
                    self.natural_hypparam + self._get_statistics(data,D)))
        # NOTE: next line is so we can use Gibbs sampling to initialize mean field
        self.mu_mf, self._sigma_mf = \
            self.mu, self.sigma * ((self.nu_mf if hasattr(self,'nu_mf') and self.nu_mf else self.nu_0) - D - 1)
        return self

    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new

    ### Mean Field

    def _resample_from_mf(self):
        self.mu, self.sigma = \
                sample_niw(*self._natural_to_standard(self.mf_natural_hypparam))
        return self

    def meanfieldupdate(self,data,weights):
        D = len(self.mu_0)
        self.mf_natural_hypparam = \
                self.natural_hypparam + self._get_weighted_statistics(data,weights,D)

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        D = len(self.mu_0)
        self.mf_natural_hypparam = \
                (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                        self.natural_hypparam
                        + 1./minibatchfrac * self._get_weighted_statistics(data,weights,D))

    @property
    def mf_natural_hypparam(self):
        return self._standard_to_natural(self.mu_mf,self.sigma_mf,self.kappa_mf,self.nu_mf)

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.mu_mf, self.sigma_mf, self.kappa_mf, self.nu_mf = \
                self._natural_to_standard(natparam)
        # NOTE: next line is for plotting
        self.mu, self.sigma = self.mu_mf, self.sigma_mf/(self.nu_mf - self.mu_mf.shape[0] - 1)

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
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde \
                - D*self.kappa_0/self.kappa_mf - self.kappa_0*self.nu_mf*\
                np.dot(self.mu_mf -
                    self.mu_0,np.linalg.solve(self.sigma_mf,self.mu_mf - self.mu_0))) \
                + invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
                + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self.nu_mf*\
                np.linalg.solve(self.sigma_mf,self.sigma_0).trace()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self.mu_mf, self.sigma_mf, self.kappa_mf, self.nu_mf
        D = len(mu_n)
        x = np.reshape(x,(-1,D)) - mu_n # x is now centered
        xs = np.linalg.solve(self.sigma_mf_chol,x.T)

        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi)

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
        # an alternative computation to the generic log_predictive, which is implemented
        # in terms of log_marginal_likelihood. mostly for testing.
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
            self.resample() # initialize from prior

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

    def resample(self,data=[]):
        self.sigma = sample_invwishart(*self._posterior_hypparams(*self._get_statistics(data)))
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
    # See Gelman's Bayesian Data Analysis notation around Eq. 3.18, p. 85 in 2nd
    # Edition. We replaced \Lambda_0 with sigma_0 since it is a prior 
    # *covariance* matrix rather than a precision matrix. This is also more
    # consistent with the notation for other Gaussians in PyBasicBayes.
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
        # It seems we should be working with lmbda and sigma inv (unless lmbda is a covariance, not a precision)
        sigma_inv, mu_0, sigma_inv_0 = self.sigma_inv, self.mu_0, self.sigma_inv_0
        if n > 0:
            sigma_inv_n = n*sigma_inv + sigma_inv_0
            mu_n = np.linalg.solve(sigma_inv_n, sigma_inv_0.dot(mu_0) + n*sigma_inv.dot(xbar))
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
        for itr in xrange(niter):
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
    def parameters(self,(mu,sigmas)):
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
            return sum((self._get_weighted_statistics(d,w) for d, w in zip(data,weights)),
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
        D = getdatadimension(data)
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

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                    self.natural_hypparam
                    + 1./minibatchfrac * self._get_weighted_statistics(data,weights))

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
            -1./2 * (np.log(betas) - special.digamma(alphas)),
            ])

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
            for itr in xrange(self.niter):
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
        mu_n, nu_n, alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        D = mu_n.shape[0]
        self.sigma = 1/np.random.gamma(alpha_n,scale=1/beta_n)
        self.mu = np.sqrt(self.sigma/nu_n)*np.random.randn(D)+mu_n
        return self

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            D = getdatadimension(data)
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
        h_0, J_0, h_mf, J_mf = self.h_0, self.J_0, self.h_mf, self.J_mf
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

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        # like meanfieldupdate except we step the factors simultaneously

        # NOTE: unlike the fully conjugate case, there are interaction terms, so
        # we work on the destructured pieces
        neff, y, ysq = self._get_weighted_statistics(data,weights)
        Emu, _ = self._E_mu
        Esigmasqinv, _ = self._E_sigmasq


        # form new natural hyperparameters as if doing a batch update
        alpha_new = self.alpha_0 + 1./minibatchfrac * 1./2*neff
        beta_new = self.beta_0 + 1./minibatchfrac * 1./2*(ysq + neff*Emu**2 - 2*Emu*y)

        h_new = self.h_0 + 1./minibatchfrac * Esigmasqinv * y
        J_new = self.J_0 + 1./minibatchfrac * Esigmasqinv * neff


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
        for niter in xrange(niter):
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
        self.mf_natural_hypparam = (self.alpha_mf, self.beta_mf, self.h_mf, self.J_mf)

        return self

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            y = weights.dot(data)
            ysq = weights.dot(data**2)
        else:
            return sum(self._get_weighted_statistics(d,w) for d,w in zip(data,weights))
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
            self.resample() # intialize from prior

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

        if high is None and not any(_ is None for _ in (x_m,alpha)):
            self.resample() # intialize from prior

    @property
    def params(self):
        return {'high':self.high}

    @property
    def hypparams(self):
        return dict(x_m=self.x_m,alpha=self.alpha,low=self.low)

    def log_likelihood(self,x):
        x = np.atleast_1d(x)
        raw = np.where((self.low <= x) & (x < self.high),
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
            datamax = max(d.max() for d in data) \
                    if n > 0 else -np.inf
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
    def __init__(self,low=None,high=None,
            x_m_low=None,alpha_low=None,x_m_high=None,alpha_high=None):
        self.low = low
        self.high = high

        self.x_m_low = x_m_low
        self.alpha_low = alpha_low
        self.x_m_high = x_m_high
        self.alpha_high = alpha_high

        if low is high is None \
                and not any(_ is None for _ in (x_m_low,alpha_low,x_m_high,alpha_high)):
            self.resample() # initialize from prior

    @property
    def params(self):
        return dict(low=self.low,high=self.high)

    @property
    def hypparams(self):
        return dict(x_m_low=self.x_m_low,alpha_low=self.alpha_low,
                x_m_high=self.x_m_high,alpha_high=self.alpha_high)

    def resample(self,data=[],niter=5):
        if len(data) == 0:
            self.low = -sample_pareto(-self.x_m_low,self.alpha_low)
            self.high = sample_pareto(self.x_m_high,self.alpha_high)
        else:
            for itr in xrange(niter):
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
            return map(self._flip_data,data)

##############
#  Discrete  #
##############

class Categorical(GibbsSampling, MeanField, MeanFieldSVI, MaxLikelihood, MAP):
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
    def __init__(self,weights=None,alpha_0=None,K=None,alphav_0=None,alpha_mf=None):
        self.K = K
        self.alpha_0 = alpha_0
        self.alphav_0 = alphav_0

        self._alpha_mf = alpha_mf if alpha_mf is not None else self.alphav_0

        self.weights = weights

        if weights is None and self.alphav_0 is not None:
            self.resample() # intialize from prior

    def _get_alpha_0(self):
        return self._alpha_0

    def _set_alpha_0(self,alpha_0):
        self._alpha_0 = alpha_0
        if not any(_ is None for _ in (self.K, self._alpha_0)):
            self.alphav_0 = np.repeat(self._alpha_0/self.K,self.K)

    alpha_0 = property(_get_alpha_0,_set_alpha_0)

    def _get_alphav_0(self):
        return self._alphav_0 if hasattr(self,'_alphav_0') else None

    def _set_alphav_0(self,alphav_0):
        if alphav_0 is not None:
            self._alphav_0 = alphav_0
            self.K = len(alphav_0)

    alphav_0 = property(_get_alphav_0,_set_alphav_0)

    @property
    def params(self):
        return dict(weights=self.weights)

    @property
    def hypparams(self):
        return dict(alphav_0=self.alphav_0)

    @property
    def num_parameters(self):
        return len(self.weights)

    def rvs(self,size=None):
        return sample_discrete(self.weights,size)

    def log_likelihood(self,x):
        return np.log(self.weights)[x]

    ### Gibbs sampling

    def resample(self,data=[],counts=None):
        counts = self._get_statistics(data,len(self.alphav_0)) \
                if counts is None else counts
        self.weights = np.random.dirichlet(np.maximum(1e-5,self.alphav_0 + counts))
        # NOTE: next line is so we can use Gibbs sampling to initialize mean field
        self._alpha_mf = self.weights * self.alphav_0.sum()
        assert (self._alpha_mf >= 0.).all()
        return self

    @staticmethod
    def _get_statistics(data,K):
        if isinstance(data,np.ndarray) or \
                (isinstance(data,list) and len(data) > 0 \
                and not isinstance(data[0],(np.ndarray,list))):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)
        return counts

    @staticmethod
    def _get_weighted_statistics(data,weights):
        # data is just a placeholder; technically it should always be
        # np.arange(K)[na,:].repeat(N,axis=0), but this code just ignores it
        if isinstance(weights,np.ndarray):
            counts = np.atleast_2d(weights).sum(0)
        else:
            counts = sum(np.atleast_2d(w).sum(0) for w in weights)
        return counts

    ### Mean Field

    def meanfieldupdate(self,data,weights):
        # update
        self._alpha_mf = self.alphav_0 + self._get_weighted_statistics(data,weights)
        self.weights = self._alpha_mf / self._alpha_mf.sum() # for plotting
        assert (self._alpha_mf > 0.).all()
        return self

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood() # default is on np.arange(self.K)
        q_entropy = -1* ((logpitilde*(self._alpha_mf-1)).sum() \
                + special.gammaln(self._alpha_mf.sum()) - special.gammaln(self._alpha_mf).sum())
        p_avgengy = special.gammaln(self.alphav_0.sum()) - special.gammaln(self.alphav_0).sum() \
                + ((self.alphav_0-1)*logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else slice(None)
        return special.digamma(self._alpha_mf[x]) - special.digamma(self._alpha_mf.sum())

    ### Mean Field SGD

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        self._alpha_mf = \
                (1-stepsize) * self._alpha_mf + stepsize * (
                        self.alphav_0
                        + 1./minibatchfrac * self._get_weighted_statistics(data,weights))
        self.weights = self._alpha_mf / self._alpha_mf.sum() # for plotting
        return self

    def _resample_from_mf(self):
        self.weights = np.random.dirichlet(self._alpha_mf)

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        K = self.K
        if weights is None:
            counts = self._get_statistics(data,K)
        else:
            counts = self._get_weighted_statistics(data,weights)
        self.weights = counts/counts.sum()
        return self

    def MAP(self,data,weights=None):
        K = self.K
        if weights is None:
            counts = self._get_statistics(data,K)
        else:
            counts = self._get_weighted_statistics(data,weights)
        counts += self.alphav_0
        self.weights = counts/counts.sum()
        return self


class CategoricalAndConcentration(Categorical):
    '''
    Categorical with resampling of the symmetric Dirichlet concentration
    parameter.

        concentration ~ Gamma(a_0,b_0)

    The Dirichlet prior over pi is then

        pi ~ Dir(concentration/K)
    '''
    def __init__(self,a_0,b_0,K,alpha_0=None,weights=None):
        self.alpha_0_obj = GammaCompoundDirichlet(a_0=a_0,b_0=b_0,K=K,concentration=alpha_0)
        super(CategoricalAndConcentration,self).__init__(alpha_0=self.alpha_0,
                K=K,weights=weights)

    def _get_alpha_0(self):
        return self.alpha_0_obj.concentration

    def _set_alpha_0(self,alpha_0):
        self.alpha_0_obj.concentration = alpha_0
        self.alphav_0 = np.repeat(alpha_0/self.K,self.K)

    alpha_0 = property(_get_alpha_0, _set_alpha_0)

    @property
    def params(self):
        return dict(alpha_0=self.alpha_0,weights=self.weights)

    @property
    def hypparams(self):
        return dict(a_0=self.a_0,b_0=self.b_0,K=self.K)

    def resample(self,data=[]):
        counts = self._get_statistics(data,self.K)
        self.alpha_0_obj.resample(counts)
        self.alpha_0 = self.alpha_0 # for the effect on alphav_0
        return super(CategoricalAndConcentration,self).resample(data)

    def resample_just_weights(self,data=[]):
        return super(CategoricalAndConcentration,self).resample(data)

    def meanfieldupdate(self,*args,**kwargs): # TODO
        warn('MeanField not implemented for %s; concentration parameter will stay fixed')
        return super(CategoricalAndConcentration,self).meanfieldupdate(*args,**kwargs)

    def max_likelihood(self,*args,**kwargs):
        raise NotImplementedError


class Multinomial(Categorical):
    '''
    Like Categorical but the data are counts, so _get_statistics is overridden
    (though _get_weighted_statistics can stay the same!). log_likelihood also
    changes since, just like for the binomial special case, we sum over all
    possible orderings.

    For example, if K == 3, then a sample with n=5 might be
        array([2,2,1])

    A Poisson process conditioned on the number of points emitted.
    '''
    def log_likelihood(self,x):
        assert isinstance(x,np.ndarray) and x.ndim == 2 and x.shape[1] == self.K
        return np.where(x,x*np.log(self.weights),0.).sum(1) \
                + special.gammaln(x.sum(1)+1) - special.gammaln(x+1).sum(1)

    def rvs(self,size=None):
        return np.bincount(super(Multinomial,self).rvs(size=size),minlength=self.K)

    @staticmethod
    def _get_statistics(data,K):
        if isinstance(data,np.ndarray):
            return np.atleast_2d(data).sum(0)
        else:
            if len(data) == 0:
                return np.zeros(K,dtype=int)
            return np.concatenate(data).sum(0)

    def expected_log_likelihood(self,x=None):
        if x is not None and (not x.ndim == 2 or not np.all(x == np.eye(x.shape[0]))):
            raise NotImplementedError # TODO nontrivial expected log likelihood
        return super(Multinomial,self).expected_log_likelihood()


class MultinomialAndConcentration(CategoricalAndConcentration,Multinomial):
    pass


class Binomial(GibbsSampling, MeanField, MeanFieldSVI):
    '''
    Models a Binomial likelihood and a Beta prior:

        p ~ Beta(alpha_0, beta_0)
        x | p ~ Binom(p,n)

    where p is the success probability, alpha_0-1 is the prior number of
    successes, beta_0-1 is the prior number of failures.

    A special case of Multinomial where N is fixed and each observation counts
    the number of successes and is in {0,1,...,N}.
    '''
    def __init__(self,alpha_0,beta_0,alpha_mf=None,beta_mf=None,p=None,n=None):
        warn('this class is untested!')
        assert n is not None

        self.n = n
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.alpha_mf = alpha_mf if alpha_mf is not None else alpha_0
        self.beta_mf = beta_mf if beta_mf is not None else beta_0

        if p is not None:
            self.p = p
        else:
            self.resample()

    def log_likelihood(self,x):
        return stats.binom.pmf(x,self.n,self.p)

    def rvs(self,size=None):
        return stats.binom.pmf(self.n,self.p,size=size)

    @property
    def natural_hypparam(self):
        return np.array([self.alpha_0 - 1, self.beta_0 - 1])

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.alpha_0, self.beta_0 = natparam + 1

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            data = data.ravel()
            tot = data.sum()
            return np.array([tot, self.n*data.shape[0] - tot])
        else:
            return sum((self._get_statistics(d) for d in data),
                    self._empty_statistics())

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            data, weights = data.ravel(), weights.ravel()
            tot = weights.dot(data)
            return np.array([tot, self.n*weights.sum() - tot])
        else:
            return sum((self._get_weighted_statistics(d,w) for d,w in zip(data,weights)),
                    self._empty_statistics())

    def _empty_statistics(self):
        return np.zeros(2)

    ### Gibbs

    def resample(self,data=[]):
        alpha_n, beta_n = self.natural_hypparam + self._get_statistics(data) + 1
        self.p = np.random.beta(alpha_n,beta_n)

        # use Gibbs to initialize mean field
        self.alpha_mf = self.p * (self.alpha_0 + self.beta_0)
        self.beta_mf = (1-self.p) * (self.alpha_0 + self.beta_0)

    ### Mean field and SVI

    def meanfieldupdate(self,data,weights):
        self.mf_natural_hypparam = \
            self.natural_hypparam + self._get_weighted_statistics(data,weights)

        # use mean field to initialize Gibbs
        self.p = self.alpha_mf / (self.alpha_mf + self.beta_mf)

    def meanfield_sgdstep(self,data,weights,minibatchprob,stepsize):
        self.mf_natural_hypparam = \
            (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                    self.natural_hypparam
                    + 1./minibatchprob * self._get_weighted_statistics(data,weights))

    @property
    def mf_natural_hypparam(self):
        return np.array([self.alpha_mf - 1, self.beta_mf - 1])

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.alpha_mf, self.beta_mf = natparam + 1

    def expected_log_likelihood(self,x):
        n = self.n
        Elnp, Eln1mp = self._mf_expected_statistics()
        return special.gammaln(n+1) - special.gammaln(x+1) - special.gammaln(n-x+1) \
                + x*Elnp + (n-x)*Eln1mp

    def _mf_expected_statistics(self):
        return special.digamma([self.alpha_mf, self.beta_mf]) \
                - special.digamma(self.alpha_mf + self.beta_mf)

    def get_vlb(self):
        Elnp, Eln1mp = self._mf_expected_statistics()
        return (self.alpha_0 - self.alpha_mf)*Elnp \
                + (self.beta_0 - self.beta_mf)*Eln1mp \
                - (self._log_partition_function(self.alpha_0, self.beta_0)
                        - self._log_partition_function(self.alpha_mf,self.beta_mf))

    def _log_partition_function(self,alpha,beta):
        return special.betaln(alpha,beta)

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


class Poisson(GibbsSampling, Collapsed, MaxLikelihood, MeanField, MeanFieldSVI):
    '''
    Poisson distribution with a conjugate Gamma prior.

    NOTE: the support is {0,1,2,...}

    Hyperparameters (following Wikipedia's notation):
        alpha_0, beta_0

    Parameter is the mean/variance parameter:
        lmbda
    '''
    def __init__(self,lmbda=None,alpha_0=None,beta_0=None,mf_alpha_0=None,mf_beta_0=None):
        self.lmbda = lmbda

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mf_alpha_0 = mf_alpha_0 if mf_alpha_0 is not None else alpha_0
        self.mf_beta_0 = mf_beta_0 if mf_beta_0 is not None else beta_0

        if lmbda is None and not any(_ is None for _ in (alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def params(self):
        return dict(lmbda=self.lmbda)

    @property
    def hypparams(self):
        return dict(alpha_0=self.alpha_0,beta_0=self.beta_0)

    def log_sf(self,x):
        return stats.poisson.logsf(x,self.lmbda)

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

    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum()
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert np.isscalar(data)
            n = 1
            tot = data

        return n, tot

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            n = weights.sum()
            tot = weights.dot(data)
        elif isinstance(data,list):
            n = sum(w.sum() for w in weights)
            tot = sum(w.dot(d) for w,d in zip(weights,data))
        else:
            assert np.isscalar(data) and np.isscalar(weights)
            n = weights
            tot = weights*data

        return np.array([n, tot])

    ### Gibbs Sampling

    def resample(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        alpha_n, beta_n = self._posterior_hypparams(*stats)
        self.lmbda = np.random.gamma(alpha_n,1/beta_n)

        # next line is for mean field initialization
        self.mf_alpha_0, self.mf_beta_0 = self.lmbda * self.beta_0, self.beta_0

        return self

    ### Mean Field

    def _resample_from_mf(self):
        mf_alpha_0, mf_beta_0 = self._natural_to_standard(self.mf_natural_hypparam)
        self.lmbda = np.random.gamma(mf_alpha_0, 1./mf_beta_0)

    def meanfieldupdate(self,data,weights):
        self.mf_natural_hypparam = \
                self.natural_hypparam + self._get_weighted_statistics(data,weights)
        self.lmbda = self.mf_alpha_0 / self.mf_beta_0

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        self.mf_natural_hypparam = \
                (1-stepsize) * self.mf_natural_hypparam + stepsize * (
                        self.natural_hypparam
                        + 1./minibatchfrac * self._get_weighted_statistics(data,weights))

    def get_vlb(self):
        return (self.natural_hypparam - self.mf_natural_hypparam).dot(self._mf_expected_statistics) \
                - (self._log_partition_fn(self.alpha_0,self.beta_0)
                        - self._log_partition_fn(self.mf_alpha_0,self.mf_beta_0))

    def expected_log_likelihood(self,x):
        Emlmbda, Elnlmbda = self._mf_expected_statistics
        return -special.gammaln(x+1) + Elnlmbda * x + Emlmbda

    @property
    def _mf_expected_statistics(self):
        alpha, beta = self.mf_alpha_0, self.mf_beta_0
        return np.array([-alpha/beta, special.digamma(alpha) - np.log(beta)])


    @property
    def natural_hypparam(self):
        return self._standard_to_natural(self.alpha_0,self.beta_0)

    @property
    def mf_natural_hypparam(self):
        return self._standard_to_natural(self.mf_alpha_0,self.mf_beta_0)

    @mf_natural_hypparam.setter
    def mf_natural_hypparam(self,natparam):
        self.mf_alpha_0, self.mf_beta_0 = self._natural_to_standard(natparam)


    def _standard_to_natural(self,alpha,beta):
        return np.array([beta, alpha-1])

    def _natural_to_standard(self,natparam):
        return natparam[1]+1, natparam[0]

    ### Collapsed

    def log_marginal_likelihood(self,data):
        return self._log_partition_fn(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_fn(self.alpha_0,self.beta_0) \
                - self._get_sum_of_gammas(data)

    def _log_partition_fn(self,alpha,beta):
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

        if n > 1e-2:
            self.lmbda = tot/n
            assert self.lmbda > 0
        else:
            self.broken = True
            self.lmbda = 999999



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
    def __init__(self,r=None,p=None,k_0=None,theta_0=None,alpha_0=None,beta_0=None):
        self.r = r
        self.p = p

        self.k_0 = k_0
        self.theta_0 = theta_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if r is p is None and not any(_ is None for _ in (k_0,theta_0,alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def params(self):
        return dict(r=self.r,p=self.p)

    @property
    def hypparams(self):
        return dict(k_0=self.k_0,theta_0=self.theta_0,
                alpha_0=self.alpha_0,beta_0=self.beta_0)

    def log_likelihood(self,x,r=None,p=None):
        r = r if r is not None else self.r
        p = p if p is not None else self.p
        x = np.array(x,ndmin=1)

        if self.p > 0:
            xnn = x[x >= 0]
            raw = np.empty(x.shape)
            raw[x>=0] = special.gammaln(r + xnn) - special.gammaln(r) \
                    - special.gammaln(xnn+1) + r*np.log(1-p) + xnn*np.log(p)
            raw[x<0] = -np.inf
            return raw if isinstance(x,np.ndarray) else raw[0]
        else:
            raw = np.log(np.zeros(x.shape))
            raw[x == 0] = 0.
            return raw if isinstance(x,np.ndarray) else raw[0]

    def log_sf(self,x):
        scalar = not isinstance(x,np.ndarray)
        x = np.atleast_1d(x)
        errs = np.seterr(divide='ignore')
        ret = np.log(special.betainc(x+1,self.r,self.p))
        np.seterr(**errs)
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
            data = np.atleast_2d(flattendata(data))
            ones = np.ones(data.shape[1],dtype=float)
            for itr in range(niter):
                ### resample r
                msum = sample_crp_tablecounts(float(self.r),data,ones).sum()
                self.r = np.random.gamma(self.k_0 + msum, 1/(1/self.theta_0 - N*np.log(1-self.p)))
                ### resample p
                self.p = np.random.beta(self.alpha_0 + data.sum(), self.beta_0 + N*self.r)
        return self

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
        return self

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
        return self

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


class NegativeBinomialFixedR(_NegativeBinomialBase, GibbsSampling, MeanField, MeanFieldSVI, MaxLikelihood):
    def __init__(self,r=None,p=None,alpha_0=None,beta_0=None,alpha_mf=None,beta_mf=None):
        self.p = p

        self.r = r

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        if p is None and not any(_ is None for _ in (alpha_0,beta_0)):
            self.resample() # intialize from prior

        if not any(_ is None for _ in (alpha_mf,beta_mf)):
            self.alpha_mf = alpha_mf
            self.beta_mf = beta_mf

    @property
    def hypparams(self):
        return dict(alpha_0=self.alpha_0,beta_0=self.beta_0)

    @property
    def natural_hypparam(self):
        return np.array([self.alpha_0,self.beta_0]) - 1

    @natural_hypparam.setter
    def natural_hypparam(self,natparam):
        self.alpha_0, self.beta_0 = natparam + 1

    ### Mean Field

    def _resample_from_mf(self):
        self.p = np.random.beta(self.alpha_mf,self.beta_mf)
        return self

    def meanfieldupdate(self,data,weights):
        self.alpha_mf, self.beta_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights))
        self.p = self.alpha_mf / (self.alpha_mf + self.beta_mf)

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        alpha_new, beta_new = \
                self._posterior_hypparams(*(
                    1./minibatchfrac * self._get_weighted_statistics(data,weights)))
        self.alpha_mf = (1-stepsize)*self.alpha_mf + stepsize*alpha_new
        self.beta_mf = (1-stepsize)*self.beta_mf + stepsize*beta_new
        self.p = self.alpha_mf / (self.alpha_mf + self.beta_mf)

    def get_vlb(self):
        Elnp, Eln1mp = self._mf_expected_statistics()
        p_avgengy = (self.alpha_0-1)*Elnp + (self.beta_0-1)*Eln1mp \
                - (special.gammaln(self.alpha_0) + special.gammaln(self.beta_0)
                        - special.gammaln(self.alpha_0 + self.beta_0))
        q_entropy = special.betaln(self.alpha_mf,self.beta_mf) \
                - (self.alpha_mf-1)*special.digamma(self.alpha_mf) \
                - (self.beta_mf-1)*special.digamma(self.beta_mf) \
                + (self.alpha_mf+self.beta_mf-2)*special.digamma(self.alpha_mf+self.beta_mf)
        return p_avgengy + q_entropy

    def _mf_expected_statistics(self):
        Elnp, Eln1mp = special.digamma([self.alpha_mf,self.beta_mf]) \
                        - special.digamma(self.alpha_mf + self.beta_mf)
        return Elnp, Eln1mp

    def expected_log_likelihood(self,x):
        Elnp, Eln1mp = self._mf_expected_statistics()
        x = np.atleast_1d(x)
        errs = np.seterr(invalid='ignore')
        out = x*Elnp + self.r*Eln1mp + self._log_base_measure(x,self.r)
        np.seterr(**errs)
        out[np.isnan(out)] = -np.inf
        return out if out.shape[0] > 1 else out[0]

    @staticmethod
    def _log_base_measure(x,r):
        return special.gammaln(x+r) - special.gammaln(x+1) - special.gammaln(r)

    ### Gibbs

    def resample(self,data=[]):
        self.p = np.random.beta(*self._posterior_hypparams(*self._get_statistics(data)))
        # set mean field params to something reasonable for initialization
        fakedata = self.rvs(10)
        self.alpha_mf, self.beta_mf = self._posterior_hypparams(*self._get_statistics(fakedata))

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)

        self.p = (tot/n) / (self.r + tot/n)
        return self

    ### Statistics and posterior hypparams

    def _get_statistics(self,data):
        if getdatasize(data) == 0:
            n, tot = 0, 0
        elif isinstance(data,np.ndarray):
            assert np.all(data >= 0)
            data = np.atleast_1d(data)
            n, tot = data.shape[0], data.sum()
        elif isinstance(data,list):
            assert all(np.all(d >= 0) for d in data)
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert np.isscalar(data)
            n = 1
            tot = data

        return np.array([n, tot])

    def _get_weighted_statistics(self,data,weights):
        if isinstance(weights,np.ndarray):
            assert np.all(data >= 0) and data.ndim == 1
            n, tot = weights.sum(), weights.dot(data)
        else:
            assert all(np.all(d >= 0) for d in data)
            n = sum(w.sum() for w in weights)
            tot = sum(w.dot(d) for d,w in zip(data,weights))

        return np.array([n, tot])

    def _posterior_hypparams(self,n,tot):
        return np.array([self.alpha_0 + tot, self.beta_0 + n*self.r])

class NegativeBinomialIntegerR2(_NegativeBinomialBase,MeanField,MeanFieldSVI,GibbsSampling):
    # NOTE: this class should replace NegativeBinomialFixedR completely...
    _fixedr_class = NegativeBinomialFixedR

    def __init__(self,alpha_0=None,beta_0=None,alphas_0=None,betas_0=None,
            r_support=None,r_probs=None,r_discrete_distn=None,
            r=None,ps=None):

        assert (r_discrete_distn is not None) ^ (r_support is not None and r_probs is not None)
        if r_discrete_distn is not None:
            r_support, = np.where(r_discrete_distn)
            r_probs = r_discrete_distn[r_support]
            r_support += 1
        self.r_support = np.asarray(r_support)
        self.rho_0 = self.rho_mf = np.log(r_probs)

        assert (alpha_0 is not None and  beta_0 is not None) \
                ^ (alphas_0 is not None and betas_0 is not None)
        alphas_0 = alphas_0 if alphas_0 is not None else [alpha_0]*len(r_support)
        betas_0 = betas_0 if betas_0 is not None else [beta_0]*len(r_support)
        ps = ps if ps is not None else [None]*len(r_support)
        self._fixedr_distns = \
            [self._fixedr_class(r=r,p=p,alpha_0=alpha_0,beta_0=beta_0)
                    for r,p,alpha_0,beta_0 in zip(r_support,ps,alphas_0,betas_0)]

        # for init
        self.ridx = sample_discrete(r_probs)
        self.r = r_support[self.ridx]

    def __repr__(self):
        return 'NB(r=%d,p=%0.3f)' % (self.r,self.p)

    @property
    def alphas_0(self):
        return np.array([d.alpha_0 for d in self._fixedr_distns]) \
                if len(self._fixedr_distns) > 0 else None

    @property
    def betas_0(self):
        return np.array([d.beta_0 for d in self._fixedr_distns]) \
                if len(self._fixedr_distns) > 0 else None

    @property
    def p(self):
        return self._fixedr_distns[self.ridx].p

    @p.setter
    def p(self,val):
        self._fixedr_distns[self.ridx].p = val

    def _resample_from_mf(self):
        self._resample_r_from_mf()
        self._resample_p_from_mf()

    def _resample_r_from_mf(self):
        lognorm = np.logaddexp.reduce(self.rho_mf)
        self.ridx = sample_discrete(np.exp(self.rho_mf - lognorm))
        self.r = self.r_support[self.ridx]

    def _resample_p_from_mf(self):
        d = self._fixedr_distns[self.ridx]
        self.p = np.random.beta(d.alpha_mf,d.beta_mf)

    def get_vlb(self):
        return self._r_vlb() + sum(np.exp(rho)*d.get_vlb()
                for rho,d in zip(self.rho_mf,self._fixedr_distns))

    def _r_vlb(self):
        return np.exp(self.rho_mf).dot(self.rho_0) \
                - np.exp(self.rho_mf).dot(self.rho_mf)

    def meanfieldupdate(self,data,weights):
        for d in self._fixedr_distns:
            d.meanfieldupdate(data,weights)
        self._update_rho_mf(data,weights)
        # everything below here is for plotting
        ridx = self.rho_mf.argmax()
        d = self._fixedr_distns[ridx]
        self.r = d.r
        self.p = d.alpha_mf / (d.alpha_mf + d.beta_mf)

    def _update_rho_mf(self,data,weights):
        self.rho_mf = self.rho_0.copy()
        for idx, d in enumerate(self._fixedr_distns):
            n, tot = d._get_weighted_statistics(data,weights)
            Elnp, Eln1mp = d._mf_expected_statistics()
            self.rho_mf[idx] += (d.alpha_0-1+tot)*Elnp + (d.beta_0-1+n*d.r)*Eln1mp
            if isinstance(data,np.ndarray):
                self.rho_mf[idx] += weights.dot(d._log_base_measure(data,d.r))
            else:
                self.rho_mf[idx] += sum(w.dot(d._log_base_measure(dt,d.r))
                        for dt,w in zip(data,weights))

    def expected_log_likelihood(self,x):
        lognorm = np.logaddexp.reduce(self.rho_mf)
        return sum(np.exp(rho-lognorm)*d.expected_log_likelihood(x)
                for rho,d in zip(self.rho_mf,self._fixedr_distns))

    def meanfield_sgdstep(self,data,weights,minibatchfrac,stepsize):
        rho_mf_orig = self.rho_mf.copy()
        if isinstance(data,np.ndarray):
            self._update_rho_mf(data,minibatchfrac*weights)
        else:
            self._update_rho_mf(data,[w*minibatchfrac for w in weights])
        rho_mf_new = self.rho_mf

        for d in self._fixedr_distns:
            d.meanfield_sgdstep(data,weights,minibatchfrac,stepsize)

        self.rho_mf = (1-stepsize)*rho_mf_orig + stepsize*rho_mf_new

        # for plotting
        ridx = self.rho_mf.argmax()
        d = self._fixedr_distns[ridx]
        self.r = d.r
        self.p = d.alpha_mf / (d.alpha_mf + d.beta_mf)

    def resample(self,data=[]):
        self._resample_r(data) # marginalizes out p values
        self._resample_p(data) # resample p given sampled r
        return self

    def _resample_r(self,data):
        self.ridx = sample_discrete(
                self._posterior_hypparams(self._get_statistics(data)))
        self.r = self.r_support[self.ridx]
        return self

    def _resample_p(self,data):
        self._fixedr_distns[self.ridx].resample(data)
        return self

    def _get_statistics(self,data=[]):
        n, tot = self._fixedr_distns[0]._get_statistics(data)
        if n > 0:
            data = flattendata(data)
            alphas_n, betas_n = self.alphas_0 + tot, self.betas_0 + self.r_support*n
            log_marg_likelihoods = \
                    special.betaln(alphas_n, betas_n) \
                        - special.betaln(self.alphas_0, self.betas_0) \
                    + (special.gammaln(data[:,na]+self.r_support)
                        - special.gammaln(data[:,na]+1) \
                        - special.gammaln(self.r_support)).sum(0)
        else:
            log_marg_likelihoods = np.zeros_like(self.r_support)
        return log_marg_likelihoods

    def _posterior_hypparams(self,log_marg_likelihoods):
        log_posterior_discrete = self.rho_0 + log_marg_likelihoods
        return np.exp(log_posterior_discrete - log_posterior_discrete.max())

class NegativeBinomialIntegerR(NegativeBinomialFixedR, GibbsSampling, MaxLikelihood):
    '''
    Nonconjugate Discrete+Beta prior
    r_discrete_distribution is an array where index i is p(r=i+1)
    '''
    def __init__(self,r_discrete_distn=None,r_support=None,
            alpha_0=None,beta_0=None,r=None,p=None):
        self.r_support = r_support
        self.r_discrete_distn = r_discrete_distn
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.r = r
        self.p = p

        if r is p is None \
                and not any(_ is None for _ in (r_discrete_distn,alpha_0,beta_0)):
            self.resample() # intialize from prior

    @property
    def hypparams(self):
        return dict(r_discrete_distn=self.r_discrete_distn,
                alpha_0=self.alpha_0,beta_0=self.beta_0)

    def get_r_discrete_distn(self):
        return self._r_discrete_distn

    def set_r_discrete_distn(self,r_discrete_distn):
        if r_discrete_distn is not None:
            r_discrete_distn = np.asarray(r_discrete_distn,dtype=np.float)
            r_support, = np.where(r_discrete_distn)
            r_probs = r_discrete_distn[r_support]
            r_probs /= r_probs.sum()
            r_support += 1 # r_probs[0] corresponds to r=1

            self.r_support = r_support
            self.r_probs = r_probs
            self._r_discrete_distn = r_discrete_distn

    r_discrete_distn = property(get_r_discrete_distn,set_r_discrete_distn)

    def rvs(self,size=None):
        out = np.random.geometric(1-self.p,size=size)-1
        for i in xrange(self.r-1):
            out += np.random.geometric(1-self.p,size=size)-1
        return out

    def resample(self,data=[]):
        alpha_n, betas_n, posterior_discrete = self._posterior_hypparams(
                *self._get_statistics(data))

        r_idx = sample_discrete(posterior_discrete)
        self.r = self.r_support[r_idx]
        self.p = np.random.beta(alpha_n, betas_n[r_idx])

    # NOTE: this class has a conjugate prior even though it's not in the
    # exponential family, so I wrote _get_statistics and _get_weighted_statistics
    # (which integrate out p) for the resample() and meanfield_update() methods,
    # though these aren't statistics in the exponential family sense

    def _get_statistics(self,data):
        # NOTE: since this isn't really in exponential family, this method needs
        # to look at hyperparameters. form posterior hyperparameters for the p
        # parameters here so we can integrate them out and get the r statistics
        n, tot = super(NegativeBinomialIntegerR,self)._get_statistics(data)
        if n > 0:
            alpha_n, betas_n = self.alpha_0 + tot, self.beta_0 + self.r_support*n
            data = flattendata(data)
            log_marg_likelihoods = \
                    special.betaln(alpha_n, betas_n) \
                        - special.betaln(self.alpha_0, self.beta_0) \
                    + (special.gammaln(data[:,na]+self.r_support)
                        - special.gammaln(data[:,na]+1) \
                        - special.gammaln(self.r_support)).sum(0)
        else:
            log_marg_likelihoods = np.zeros_like(self.r_support)

        return n, tot, log_marg_likelihoods

    def _get_weighted_statistics(self,data,weights):
        n, tot = super(NegativeBinomialIntegerR,self)._get_weighted_statistics(data,weights)
        if n > 0:
            alpha_n, betas_n = self.alpha_0 + tot, self.beta_0 + self.r_support*n
            data, weights = flattendata(data), flattendata(weights)
            log_marg_likelihoods = \
                    special.betaln(alpha_n, betas_n) \
                        - special.betaln(self.alpha_0, self.beta_0) \
                    + (special.gammaln(data[:,na]+self.r_support)
                        - special.gammaln(data[:,na]+1) \
                        - special.gammaln(self.r_support)).dot(weights)
        else:
            log_marg_likelihoods = np.zeros_like(self.r_support)

        return n, tot, log_marg_likelihoods

    def _posterior_hypparams(self,n,tot,log_marg_likelihoods):
        alpha_n = self.alpha_0 + tot
        betas_n = self.beta_0 + n*self.r_support
        log_posterior_discrete = np.log(self.r_probs) + log_marg_likelihoods
        posterior_discrete = np.exp(log_posterior_discrete - log_posterior_discrete.max())
        return alpha_n, betas_n, posterior_discrete

    def max_likelihood(self,data,weights=None,stats=None):
        if stats is not None:
            n, tot = stats
        elif weights is None:
            n, tot = super(NegativeBinomialIntegerR,self)._get_statistics(data)
        else:
            n, tot = super(NegativeBinomialIntegerR,self)._get_weighted_statistics(data,weights)

        if n > 1:
            rs = self.r_support
            ps = self._max_likelihood_ps(n,tot,rs)

            # TODO TODO this isn't right for weighted data: do weighted sums
            if isinstance(data,np.ndarray):
                likelihoods = np.array([self.log_likelihood(data,r=r,p=p).sum()
                                            for r,p in zip(rs,ps)])
            else:
                likelihoods = np.array([sum(self.log_likelihood(d,r=r,p=p).sum()
                                            for d in data) for r,p in zip(rs,ps)])

            argmax = likelihoods.argmax()
            self.r = self.r_support[argmax]
            self.p = ps[argmax]
        return self

    def _log_base_measure(self,data):
        return [(special.gammaln(r+data) - special.gammaln(r) - special.gammaln(data+1)).sum()
                for r in self.r_support]

    def _max_likelihood_ps(self,n,tot,rs):
        ps = (tot/n) / (rs + tot/n)
        assert (ps >= 0).all()
        return ps

class _StartAtRMixin(object):
    def log_likelihood(self,x,**kwargs):
        r = kwargs['r'] if 'r' in kwargs else self.r
        return super(_StartAtRMixin,self).log_likelihood(x-r,**kwargs)

    def log_sf(self,x,**kwargs):
        return super(_StartAtRMixin,self).log_sf(x-self.r,**kwargs)

    def expected_log_likelihood(self,x,**kwargs):
        r = kwargs['r'] if 'r' in kwargs else self.r
        return super(_StartAtRMixin,self).expected_log_likelihood(x-r,**kwargs)

    def rvs(self,size=[]):
        return super(_StartAtRMixin,self).rvs(size)+self.r

class NegativeBinomialFixedRVariant(_StartAtRMixin,NegativeBinomialFixedR):
    def _get_statistics(self,data):
        n, tot = super(NegativeBinomialFixedRVariant,self)._get_statistics(data)
        n, tot = n, tot-n*self.r
        assert tot >= 0
        return np.array([n, tot])

    def _get_weighted_statistics(self,data,weights):
        n, tot = super(NegativeBinomialFixedRVariant,self)._get_weighted_statistics(data,weights)
        n, tot = n, tot-n*self.r
        assert tot >= 0
        return np.array([n, tot])

class NegativeBinomialIntegerRVariant(NegativeBinomialIntegerR):
    def resample(self,data=[]):
        n, alpha_n, posterior_discrete, r_support = self._posterior_hypparams(
                *self._get_statistics(data)) # NOTE: pass out r_support b/c feasible subset
        self.r = r_support[sample_discrete(posterior_discrete)]
        self.p = np.random.beta(alpha_n - n*self.r, self.beta_0 + n*self.r)

    def _get_statistics(self,data):
        n = getdatasize(data)
        if n > 0:
            data = flattendata(data)
            feasible = self.r_support <= data.min()
            assert np.any(feasible)
            r_support = self.r_support[feasible]
            normalizers = (special.gammaln(data[:,na]) - special.gammaln(data[:,na]-r_support+1)
                    - special.gammaln(r_support)).sum(0)
            return n, data.sum(), normalizers, feasible
        else:
            return n, None, None, None

    def _posterior_hypparams(self,n,tot,normalizers,feasible):
        if n == 0:
            return n, self.alpha_0, self.r_probs, self.r_support
        else:
            r_probs = self.r_probs[feasible]
            r_support = self.r_support[feasible]
            log_marg_likelihoods = special.betaln(self.alpha_0 + tot - n*r_support,
                                                        self.beta_0 + r_support*n) \
                                    - special.betaln(self.alpha_0, self.beta_0) \
                                    + normalizers
            log_marg_probs = np.log(r_probs) + log_marg_likelihoods
            log_marg_probs -= log_marg_probs.max()
            marg_probs = np.exp(log_marg_probs)

            return n, self.alpha_0 + tot, marg_probs, r_support

    def _max_likelihood_ps(self,n,tot,rs):
        ps = 1-(rs*n)/tot
        assert (ps >= 0).all()
        return ps

    def rvs(self,size=[]):
        return super(NegativeBinomialIntegerRVariant,self).rvs(size) + self.r

class NegativeBinomialIntegerR2Variant(NegativeBinomialIntegerR2):
    _fixedr_class = NegativeBinomialFixedRVariant

    def _update_rho_mf(self,data,weights):
        self.rho_mf = self.rho_0.copy()
        for idx, d in enumerate(self._fixedr_distns):
            n, tot = d._get_weighted_statistics(data,weights)
            Elnp, Eln1mp = d._mf_expected_statistics()
            self.rho_mf[idx] += (d.alpha_0-1+tot)*Elnp + (d.beta_0-1+n*d.r)*Eln1mp
            self.rho_mf_temp = self.rho_mf.copy()

            # NOTE: this method only needs to override parent in the base measure
            # part, i.e. data -> data-r
            if isinstance(data,np.ndarray):
                self.rho_mf[idx] += weights.dot(d._log_base_measure(data-d.r,d.r))
            else:
                self.rho_mf[idx] += sum(w.dot(d._log_base_measure(dt-d.r,d.r))
                        for dt,w in zip(data,weights))

class CRP(GibbsSampling):
    '''
    concentration ~ Gamma(a_0,b_0) [b_0 is inverse scale, inverse of numpy scale arg]
    rvs ~ CRP(concentration)

    This class models CRPs. The parameter is the concentration parameter (proportional
    to probability of starting a new table given some number of customers in the
    restaurant), which has a Gamma prior.
    '''

    def __init__(self,a_0,b_0,concentration=None):
        self.a_0 = a_0
        self.b_0 = b_0

        if concentration is not None:
            self.concentration = concentration
        else:
            self.resample(niter=1)

    @property
    def params(self):
        return dict(concentration=self.concentration)

    @property
    def hypparams(self):
        return dict(a_0=self.a_0,b_0=self.b_0)

    def rvs(self,customer_counts):
        # could replace this with one of the faster C versions I have lying
        # around, but at least the Python version is clearer
        assert isinstance(customer_counts,list) or isinstance(customer_counts,int)
        if isinstance(customer_counts,int):
            customer_counts = [customer_counts]

        restaurants = []
        for num in customer_counts:
            # a CRP with num customers
            tables = []
            for c in range(num):
                newidx = sample_discrete(np.array(tables + [self.concentration]))
                if newidx == len(tables):
                    tables += [1]
                else:
                    tables[newidx] += 1

            restaurants.append(tables)

        return restaurants if len(restaurants) > 1 else restaurants[0]

    def log_likelihood(self,restaurants):
        assert isinstance(restaurants,list) and len(restaurants) > 0
        if not isinstance(restaurants[0],list): restaurants=[restaurants]

        likes = []
        for counts in restaurants:
            counts = np.array([c for c in counts if c > 0])    # remove zero counts b/c of gammaln
            K = len(counts) # number of tables
            N = sum(counts) # number of customers
            likes.append(K*np.log(self.concentration) + np.sum(special.gammaln(counts)) +
                            special.gammaln(self.concentration) -
                            special.gammaln(N+self.concentration))

        return np.asarray(likes) if len(likes) > 1 else likes[0]

    def resample(self,data=[],niter=50):
        for itr in range(niter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(data))
            self.concentration = np.random.gamma(a_n,scale=1./b_n)

    def _posterior_hypparams(self,sample_numbers,total_num_distinct):
        # NOTE: this is a stochastic function: it samples auxiliary variables
        if total_num_distinct > 0:
            sample_numbers = np.array(sample_numbers)
            sample_numbers = sample_numbers[sample_numbers > 0]

            wvec = np.random.beta(self.concentration+1,sample_numbers)
            svec = np.array(stats.bernoulli.rvs(sample_numbers/(sample_numbers+self.concentration)))
            return self.a_0 + total_num_distinct-svec.sum(), (self.b_0 - np.log(wvec).sum())
        else:
            return self.a_0, self.b_0
        return self

    def _get_statistics(self,data):
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

class GammaCompoundDirichlet(CRP):
    # TODO this class is a bit ugly
    '''
    Implements a Gamma(a_0,b_0) prior over finite dirichlet concentration
    parameter. The concentration is scaled according to the weak-limit sequence.

    For each set of counts i, the model is
        concentration ~ Gamma(a_0,b_0)
        pi_i ~ Dir(concentration/K)
        data_i ~ Multinomial(pi_i)

    K is a free parameter in that with big enough K (relative to the size of the
    sampled data) everything starts to act like a DP; K is just the size of the
    size of the mesh projection.
    '''
    def __init__(self,K,a_0,b_0,concentration=None):
        self.K = K
        super(GammaCompoundDirichlet,self).__init__(a_0=a_0,b_0=b_0,
                concentration=concentration)

    @property
    def params(self):
        return dict(concentration=self.concentration)

    @property
    def hypparams(self):
        return dict(a_0=self.a_0,b_0=self.b_0,K=self.K)

    def rvs(self,sample_counts):
        if isinstance(sample_counts,int):
            sample_counts = [sample_counts]
        out = np.empty((len(sample_counts),self.K),dtype=int)
        for idx,c in enumerate(sample_counts):
            out[idx] = np.random.multinomial(c,
                np.random.dirichlet(np.repeat(self.concentration/self.K,self.K)))
        return out if out.shape[0] > 1 else out[0]

    def resample(self,data=[],niter=50,weighted_cols=None):
        if weighted_cols is not None:
            self.weighted_cols = weighted_cols
        else:
            self.weighted_cols = np.ones(self.K)

        # all this is to check if data is empty
        if isinstance(data,np.ndarray):
            size = data.sum()
        elif isinstance(data,list):
            size = sum(d.sum() for d in data)
        else:
            assert data == 0
            size = 0

        if size > 0:
            return super(GammaCompoundDirichlet,self).resample(data,niter=niter)
        else:
            return super(GammaCompoundDirichlet,self).resample(data,niter=1)

    def _get_statistics(self,data):
        # NOTE: this is a stochastic function: it samples auxiliary variables
        counts = np.array(data,ndmin=2,order='C')

        # sample m's, which sample an inverse of the weak limit projection
        if counts.sum() == 0:
            return 0, 0
        else:
            m = sample_crp_tablecounts(self.concentration,counts,self.weighted_cols)
            return counts.sum(1), m.sum()

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

