from __future__ import division
from builtins import zip
from builtins import map
from builtins import range
__all__ = ['Categorical', 'CategoricalAndConcentration', 'Multinomial',
           'MultinomialAndConcentration', 'GammaCompoundDirichlet', 'CRP']

import numpy as np
from warnings import warn
import scipy.stats as stats
import scipy.special as special

from pybasicbayes.abstractions import \
    GibbsSampling, MeanField, MeanFieldSVI, MaxLikelihood, MAP

from pybasicbayes.util.stats import sample_discrete

try:
    from pybasicbayes.util.cstats import sample_crp_tablecounts
except ImportError:
    warn('using slow sample_crp_tablecounts')
    from pybasicbayes.util.stats import sample_crp_tablecounts


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
            self.resample()  # intialize from prior

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
        out = np.zeros_like(x, dtype=np.double)
        nanidx = np.isnan(x)
        err = np.seterr(divide='ignore')
        out[~nanidx] = np.log(self.weights)[list(x[~nanidx])]  # log(0) can happen, no warning
        np.seterr(**err)
        return out

    ### Gibbs sampling

    def resample(self,data=[],counts=None):
        counts = self._get_statistics(data) if counts is None else counts
        self.weights = np.random.dirichlet(self.alphav_0 + counts)
        np.clip(self.weights, np.spacing(1.), np.inf, out=self.weights)
        # NOTE: next line is so we can use Gibbs sampling to initialize mean field
        self._alpha_mf = self.weights * self.alphav_0.sum()
        assert (self._alpha_mf >= 0.).all()
        return self

    def _get_statistics(self,data,K=None):
        K = K if K else self.K
        if isinstance(data,np.ndarray) or \
                (isinstance(data,list) and len(data) > 0
                 and not isinstance(data[0],(np.ndarray,list))):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)
        return counts

    def _get_weighted_statistics(self,data,weights):
        if isinstance(weights,np.ndarray):
            assert weights.ndim in (1,2)
            if data is None or weights.ndim == 2:
                # when weights is 2D or data is None, the weights are expected
                # indicators and data is just a placeholder; nominally data
                # should be np.arange(K)[na,:].repeat(N,axis=0)
                counts = np.atleast_2d(weights).sum(0)
            else:
                # when weights is 1D, data is indices and we do a weighted
                # bincount
                counts = np.bincount(data,weights,minlength=self.K)
        else:
            if len(weights) == 0:
                counts = np.zeros(self.K,dtype=int)
            else:
                data = data if data else [None]*len(weights)
                counts = sum(self._get_weighted_statistics(d,w)
                             for d, w in zip(data,weights))
        return counts

    ### Mean Field

    def meanfieldupdate(self,data,weights):
        # update
        self._alpha_mf = self.alphav_0 + self._get_weighted_statistics(data,weights)
        self.weights = self._alpha_mf / self._alpha_mf.sum()  # for plotting
        assert (self._alpha_mf > 0.).all()
        return self

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood()  # default is on np.arange(self.K)
        q_entropy = -1* (
            (logpitilde*(self._alpha_mf-1)).sum()
            + special.gammaln(self._alpha_mf.sum()) - special.gammaln(self._alpha_mf).sum())
        p_avgengy = special.gammaln(self.alphav_0.sum()) - special.gammaln(self.alphav_0).sum() \
            + ((self.alphav_0-1)*logpitilde).sum()

        return p_avgengy + q_entropy

    def expected_log_likelihood(self,x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else slice(None)
        return special.digamma(self._alpha_mf[x]) - special.digamma(self._alpha_mf.sum())

    ### Mean Field SGD

    def meanfield_sgdstep(self,data,weights,prob,stepsize):
        self._alpha_mf = \
            (1-stepsize) * self._alpha_mf + stepsize * (
                self.alphav_0
                + 1./prob * self._get_weighted_statistics(data,weights))
        self.weights = self._alpha_mf / self._alpha_mf.sum()  # for plotting
        return self

    def _resample_from_mf(self):
        self.weights = np.random.dirichlet(self._alpha_mf)

    ### Max likelihood

    def max_likelihood(self,data,weights=None):
        if weights is None:
            counts = self._get_statistics(data)
        else:
            counts = self._get_weighted_statistics(data,weights)
        self.weights = counts/counts.sum()
        return self

    def MAP(self,data,weights=None):
        if weights is None:
            counts = self._get_statistics(data)
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
        self.alpha_0 = self.alpha_0  # for the effect on alphav_0
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
    def __init__(self,weights=None,alpha_0=None,K=None,alphav_0=None,alpha_mf=None,
                 N=1):
        self.N = N
        super(Multinomial, self).__init__(weights,alpha_0,K,alphav_0,alpha_mf)

    def log_likelihood(self,x):
        assert isinstance(x,np.ndarray) and x.ndim == 2 and x.shape[1] == self.K
        return np.where(x,x*np.log(self.weights),0.).sum(1) \
            + special.gammaln(x.sum(1)+1) - special.gammaln(x+1).sum(1)

    def rvs(self,size=None,N=None):
        N = N if N else self.N
        return np.random.multinomial(N, self.weights, size=size)

    def _get_statistics(self,data,K=None):
        K = K if K else self.K
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
                sample_numbers = np.array(list(map(sum,data)))
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

    def rvs(self, sample_counts=None, size=None):
        if sample_counts is None:
            sample_counts = size
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

