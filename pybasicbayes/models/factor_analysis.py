"""
Probabilistic factor analysis to perform dimensionality reduction on mouse images.
With the probabilistic approach, we can handle missing data in the images.
Technically this holds for missing at random data, but we can try it
out on images where we treat cable pixels as missing, even though they
won't be random. This should give us a model-based way to fill in pixels,
and hopefully a more robust way to estimate principle components for modeling.
"""
import abc
import numpy as np

from pybasicbayes.abstractions import Model, \
    ModelGibbsSampling, ModelMeanField, ModelMeanFieldSVI, ModelEM
from pybasicbayes.util.stats import sample_gaussian
from pybasicbayes.util.general import objarray

from pybasicbayes.distributions import DiagonalRegression

from pybasicbayes.util.profiling import line_profiled
PROFILING = True

class FactorAnalysisStates(object):
    """
    Wrapper for the latent states of a factor analysis model
    """
    def __init__(self, model, data, mask=None):
        self.model = model
        self.X = data
        self.mask = mask
        if mask is None:
            mask = np.ones_like(data, dtype=bool)
        assert data.shape == mask.shape and mask.dtype == bool
        assert self.X.shape[1] == self.D_obs

        # Initialize latent states
        self.N = self.X.shape[0]
        self.Z = np.random.randn(self.N, self.D_latent)

    @property
    def D_obs(self):
        return self.model.D_obs

    @property
    def D_latent(self):
        return self.model.D_latent

    @property
    def W(self):
        return self.model.W

    @property
    def sigmasq(self):
        return self.model.sigmasq

    @property
    def regression(self):
        return self.model.regression


    def log_likelihood(self):
        mu = np.dot(self.Z, self.W.T)
        return -0.5 * np.sum(((self.X - mu) * self.mask) ** 2 / self.sigmasq)

    ## Gibbs
    def resample(self):
        W, sigmasq = self.W, self.sigmasq
        J0 = np.eye(self.D_latent)
        h0 = np.zeros(self.D_latent)

        # Sample each latent embedding
        for n in range(self.N):
            Jobs = self.mask[n] / sigmasq
            Jpost = J0 + (W * Jobs[:, None]).T.dot(W)
            hpost = h0 + (self.X[n] * Jobs).dot(W)
            self.Z[n] = sample_gaussian(J=Jpost, h=hpost)

    ## Mean field
    def E_step(self):
        W = self.W
        WWT = np.array([np.outer(wd,wd) for wd in W])
        sigmasq_inv = 1./self.sigmasq
        self._meanfieldupdate(W, WWT, sigmasq_inv)

        # Copy over the expected states to Z
        self.Z = self.E_Z

    def meanfieldupdate(self):
        E_W, E_WWT, E_sigmasq_inv, _ = self.regression.mf_expectations
        self._meanfieldupdate(E_W, E_WWT, E_sigmasq_inv)

    def _meanfieldupdate(self, E_W, E_WWT, E_sigmasq_inv):
        N, D_obs, D_lat = self.N, self.D_obs, self.D_latent
        E_WWT_vec = E_WWT.reshape(D_obs, -1)

        J0 = np.eye(D_lat)
        h0 = np.zeros(D_lat)

        # Get expectations for the latent embedding of these datapoints
        self.E_Z = np.zeros((N, D_lat))
        self.E_ZZT = np.zeros((N, D_lat, D_lat))

        for n in range(N):
            Jobs = self.mask[n] * E_sigmasq_inv
            # Faster than Jpost = J0 + np.sum(E_WWT * Jobs[:,None,None], axis=0)
            Jpost = J0 + (np.dot(Jobs, E_WWT_vec)).reshape((D_lat, D_lat))
            hpost = h0 + (self.X[n] * Jobs).dot(E_W)

            # Get the expectations for this set of indices
            Sigma_post = np.linalg.inv(Jpost)
            self.E_Z[n] = Sigma_post.dot(hpost)
            self.E_ZZT[n] = Sigma_post + np.outer(self.E_Z[n], self.E_Z[n])

        self._set_expected_stats()

    def _set_expected_stats(self):
        D_lat = self.D_latent
        E_Xsq = np.sum(self.X**2 * self.mask, axis=0)
        E_XZT = (self.X * self.mask).T.dot(self.E_Z)
        E_ZZT_vec = self.E_ZZT.reshape((self.E_ZZT.shape[0], D_lat ** 2))
        E_ZZT = np.array([np.dot(self.mask[:, d], E_ZZT_vec).reshape((D_lat, D_lat))
                          for d in range(self.D_obs)])
        n = np.sum(self.mask, axis=0)

        self.E_emission_stats = objarray([E_Xsq, E_XZT, E_ZZT, n])

    def resample_from_mf(self):
        for n in range(self.N):
            mu_n = self.E_Z[n]
            Sigma_n = self.E_ZZT[n] - np.outer(mu_n, mu_n)
            self.Z[n] = sample_gaussian(mu=mu_n, Sigma=Sigma_n)

    def expected_log_likelihood(self):
        E_W, E_WWT, E_sigmasq_inv, E_log_sigmasq = self.regression.mf_expectations
        E_Xsq, E_XZT, E_ZZT, n = self.E_emission_stats

        ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.sum(E_log_sigmasq * self.mask)
        ll += -0.5 * np.sum(E_Xsq * E_sigmasq_inv)
        ll += -0.5 * np.sum(-2 * E_XZT * E_W * E_sigmasq_inv[:,None])
        ll += -0.5 * np.sum(E_WWT * E_ZZT * E_sigmasq_inv[:,None,None])
        return ll


class _FactorAnalysisBase(Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self, D_obs, D_latent,
                 W=None, sigmasq=None,
                 sigmasq_W_0=1.0, mu_W_0=0.0,
                 alpha_0=3.0, beta_0=2.0):

        self.D_obs, self.D_latent = D_obs, D_latent

        # The weights and variances are encapsulated in a DiagonalRegression class
        self.regression = \
            DiagonalRegression(
                self.D_obs, self.D_latent,
                mu_0=mu_W_0 * np.ones(self.D_latent),
                Sigma_0=sigmasq_W_0 * np.eye(self.D_latent),
                alpha_0=alpha_0, beta_0=beta_0,
                A=W, sigmasq=sigmasq)

        self.data_list = []

    @property
    def W(self):
        return self.regression.A

    @property
    def sigmasq(self):
        return self.regression.sigmasq_flat

    def add_data(self, data, mask=None):
        self.data_list.append(FactorAnalysisStates(self, data, mask=mask))
        return self.data_list[-1]

    def generate(self, keep=True, N=1, mask=None, **kwargs):
        # Sample from the factor analysis model
        W, sigmasq = self.W, self.sigmasq
        Z = np.random.randn(N, self.D_latent)
        X = np.dot(Z, W.T) + np.sqrt(sigmasq) * np.random.randn(N, self.D_obs)

        data = FactorAnalysisStates(self, X, mask=mask)
        data.Z = Z
        if keep:
            self.data_list.append(data)
        return data

    def log_likelihood(self):
        return np.sum([d.log_likelihood() for d in self.data_list])

    def log_probability(self):
        lp = 0

        # Prior
        # lp += (-self.alpha_0-1) * np.log(self.sigmasq) - self.beta_0 / self.sigmasq
        lp += -0.5 * np.sum(self.W**2)
        lp += -0.5 * np.sum(self.Z**2)
        lp += self.log_likelihood()
        return lp


class _FactorAnalysisGibbs(ModelGibbsSampling, _FactorAnalysisBase):
    __metaclass__ = abc.ABCMeta

    def resample_model(self):
        for data in self.data_list:
            data.resample()

        Zs = np.vstack([d.Z for d in self.data_list])
        Xs = np.vstack([d.X for d in self.data_list])
        mask = np.vstack([d.mask for d in self.data_list])
        self.regression.resample((Zs, Xs), mask=mask)


class _FactorAnalysisEM(ModelEM, _FactorAnalysisBase):

    def _null_stats(self):
        return objarray(
            [np.zeros(self.D_obs),
             np.zeros((self.D_obs, self.D_latent)),
             np.zeros((self.D_obs, self.D_latent, self.D_latent)),
             np.zeros(self.D_obs)])

    def log_likelihood(self):
        # TODO: Fix inheritance issues...
        return np.sum([d.log_likelihood() for d in self.data_list])

    def EM_step(self):
        for data in self.data_list:
            data.E_step()

        stats = self._null_stats() + sum([d.E_emission_stats for d in self.data_list])
        self.regression.max_likelihood(data=None, weights=None, stats=stats)


class _FactorAnalysisMeanField(ModelMeanField, ModelMeanFieldSVI, _FactorAnalysisBase):
    __metaclass__ = abc.ABCMeta

    def _null_stats(self):
        return objarray(
            [np.zeros(self.D_obs),
             np.zeros((self.D_obs, self.D_latent)),
             np.zeros((self.D_obs, self.D_latent, self.D_latent)),
             np.zeros(self.D_obs)])

    def meanfield_coordinate_descent_step(self):
        for data in self.data_list:
            data.meanfieldupdate()

        stats = self._null_stats() + sum([d.E_emission_stats for d in self.data_list])
        self.regression.meanfieldupdate(stats=stats)

    def meanfield_sgdstep(self, minibatch, prob, stepsize, masks=None):
        assert stepsize > 0 and stepsize <= 1

        states_list = self._get_mb_states_list(minibatch, masks)
        for s in states_list:
            s.meanfieldupdate()

        # Compute the sufficient statistics of the latent parameters
        self.regression.meanfield_sgdstep(
            data=None, weights=None, prob=prob, stepsize=stepsize,
            stats=(sum(s.E_emission_stats for s in states_list)))

        # Compute the expected log likelihood for this minibatch
        return sum([s.expected_log_likelihood() for s in states_list])

    def _get_mb_states_list(self, minibatch, masks):
        minibatch = minibatch if isinstance(minibatch, list) else [minibatch]
        masks = [None] * len(minibatch) if masks is None else \
            (masks if isinstance(masks, list) else [masks])

        def get_states(data, mask):
            self.add_data(data, mask=mask)
            return self.data_list.pop()

        return [get_states(data, mask) for data, mask in zip(minibatch, masks)]

    def resample_from_mf(self):
        for data in self.data_list:
            data.resample_from_mf()
        self.regression.resample_from_mf()

    def expected_log_likelihood(self):
        ell = 0
        for data in self.data_list:
            ell += data.expected_log_likelihood()
        return ell

    def initialize_meanfield(self):
        self.regression._initialize_mean_field()


class FactorAnalysis(_FactorAnalysisGibbs, _FactorAnalysisEM, _FactorAnalysisMeanField):
    pass

