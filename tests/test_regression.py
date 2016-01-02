from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, MaxLikelihoodTester, \
    GewekeGibbsTester, BasicTester


@attr('regression')
class TestRegression(
        BasicTester,BigDataGibbsTester,
        MaxLikelihoodTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Regression

    @property
    def hyperparameter_settings(self):
        return (dict(nu_0=3,S_0=np.eye(1),M_0=np.zeros((1,2)),K_0=np.eye(2)),
                dict(nu_0=5,S_0=np.eye(2),M_0=np.zeros((2,4)),K_0=2*np.eye(4)),
                dict(nu_0=5,S_0=np.eye(2),M_0=np.zeros((2,5)),K_0=2*np.eye(5),affine=True),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.A-d2.A) < 0.1 and np.linalg.norm(d1.sigma-d2.sigma) < 0.1

    @property
    def big_data_size(self):
        return 80000

    def geweke_statistics(self,d,data):
        return np.concatenate((d.A.flatten(),np.diag(d.sigma)))

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.A.flatten().shape[0])

    @property
    def geweke_ntrials(self):
        return 1  # because it's slow

    @property
    def geweke_num_statistic_fails_to_tolerate(self):
        return 0

    ### class-specific

    def test_affine_loglike(self):
        A = np.random.randn(2,3)
        b = np.random.randn(2)
        sigma = np.random.randn(2,2); sigma = sigma.dot(sigma.T)
        data = np.random.randn(25,5)

        d1 = self.distribution_class(A=np.hstack((A,b[:,None])),sigma=sigma,affine=True)
        d2 = self.distribution_class(A=A,sigma=sigma)

        likes1 = d1.log_likelihood(data)
        data[:,-2:] -= b
        likes2 = d2.log_likelihood(data)

        assert np.allclose(likes1,likes2)

    def test_loglike_against_gaussian(self):
        mu = np.random.randn(3)
        A = mu[:,None]
        sigma = np.random.randn(3,3); sigma = sigma.dot(sigma.T)

        data = np.random.randn(25,mu.shape[0])

        d1 = distributions.Gaussian(mu=mu,sigma=sigma)
        likes1 = d1.log_likelihood(data)

        d2 = self.distribution_class(A=A,sigma=sigma)
        likes2 = d2.log_likelihood(np.hstack((np.ones((data.shape[0],1)),data)))

        assert np.allclose(likes1,likes2)

@attr('regressionnonconj')
class TestRegressionNonconj(BasicTester,BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.RegressionNonconj

    @property
    def hyperparameter_settings(self):
        def make_hyps(m,n):
            return dict(nu_0=m+1, S_0=m*np.eye(m),
                        M_0=np.zeros((m,n)), Sigma_0=np.eye(m*n))
        return [make_hyps(m,n) for m, n in [(2,3), (3,2)]]

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.A-d2.A) < 0.5 and np.linalg.norm(d1.sigma-d2.sigma) < 0.5

    def geweke_statistics(self,d,data):
        return np.concatenate((d.A.flatten(),np.diag(d.sigma)))

    def geweke_numerical_slices(self,d,setting_idx):
        return slice(0,d.A.flatten().shape[0])

    @property
    def geweke_ntrials(self):
        return 1  # because it's slow

    @property
    def geweke_num_statistic_fails_to_tolerate(self):
        return 0

    @property
    def geweke_resample_kwargs(self):
        return dict(niter=2)
