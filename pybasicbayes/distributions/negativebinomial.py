from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
__all__ = [
    'NegativeBinomial', 'NegativeBinomialFixedR', 'NegativeBinomialIntegerR2',
    'NegativeBinomialIntegerR', 'NegativeBinomialFixedRVariant',
    'NegativeBinomialIntegerRVariant', 'NegativeBinomialIntegerRVariant',
    'NegativeBinomialIntegerR2Variant']

import numpy as np
from numpy import newaxis as na
import scipy.special as special
from scipy.misc import logsumexp
from warnings import warn

from pybasicbayes.abstractions import Distribution, GibbsSampling, \
    MeanField, MeanFieldSVI, MaxLikelihood
from pybasicbayes.util.stats import getdatasize, flattendata, \
    sample_discrete_from_log, sample_discrete, atleast_2d

try:
    from pybasicbayes.util.cstats import sample_crp_tablecounts
except ImportError:
    warn('using slow sample_crp_tablecounts')
    from pybasicbayes.util.stats import sample_crp_tablecounts


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
            data = atleast_2d(flattendata(data))
            N = len(data)
            for itr in range(niter):
                ### resample r
                msum = sample_crp_tablecounts(self.r,data).sum()
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

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        alpha_new, beta_new = \
                self._posterior_hypparams(*(
                    1./prob * self._get_weighted_statistics(data,weights)))
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
        lognorm = logsumexp(self.rho_mf)
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
        lognorm = logsumexp(self.rho_mf)
        return sum(np.exp(rho-lognorm)*d.expected_log_likelihood(x)
                for rho,d in zip(self.rho_mf,self._fixedr_distns))

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        rho_mf_orig = self.rho_mf.copy()
        if isinstance(data,np.ndarray):
            self._update_rho_mf(data,prob*weights)
        else:
            self._update_rho_mf(data,[w*prob for w in weights])
        rho_mf_new = self.rho_mf

        for d in self._fixedr_distns:
            d.meanfield_sgdstep(data,weights,prob,stepsize)

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
        for i in range(self.r-1):
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
