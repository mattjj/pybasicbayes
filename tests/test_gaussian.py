from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, \
    GewekeGibbsTester, BasicTester


@attr('gaussian')
class TestGaussian(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Gaussian

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=1.,nu_0=4.),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1 and np.linalg.norm(d1.sigma-d2.sigma) < 0.1

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,np.diag(d.sigma)))

    @property
    def geweke_nsamples(self):
        return 50000

    @property
    def geweke_data_size(self):
        return 1

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.mu.shape[0])

    ### class-specific

    def test_empirical_bayes(self):
        data = np.random.randn(50,2)
        distributions.Gaussian().empirical_bayes(data).hypparams


@attr('diagonalgaussian')
class TestDiagonalGaussian(BigDataGibbsTester,GewekeGibbsTester,BasicTester):
    @property
    def distribution_class(self):
        return distributions.DiagonalGaussian

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=np.zeros(2),nus_0=7,alphas_0=np.r_[5.,10.],betas_0=np.r_[1.,4.]),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1 and np.linalg.norm(d1.sigmas-d2.sigmas) < 0.25

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,d.sigmas))

    @property
    def geweke_nsamples(self):
        return 50000

    @property
    def geweke_data_size(self):
        return 2

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.mu.shape[0])

    ### class-specific

    def test_log_likelihood(self):
        data = np.random.randn(1000,100)

        mu = np.random.randn(100)
        sigmas = np.random.uniform(1,2,size=100)

        d = distributions.DiagonalGaussian(mu=mu,sigmas=sigmas)
        pdf1 = d.log_likelihood(data)

        import scipy.stats as stats
        pdf2 = stats.norm.logpdf(data,loc=mu,scale=np.sqrt(sigmas)).sum(1)

        assert np.allclose(pdf1,pdf2)

    def test_log_likelihood2(self):
        data = np.random.randn(1000,600)

        mu = np.random.randn(600)
        sigmas = np.random.uniform(1,2,size=600)

        d = distributions.DiagonalGaussian(mu=mu,sigmas=sigmas)
        pdf1 = d.log_likelihood(data)

        import scipy.stats as stats
        pdf2 = stats.norm.logpdf(data,loc=mu,scale=np.sqrt(sigmas)).sum(1)

        assert np.allclose(pdf1,pdf2)


@attr('diagonalgaussiannonconj')
class TestDiagonalGaussianNonconjNIG(BigDataGibbsTester,GewekeGibbsTester,BasicTester):
    @property
    def distribution_class(self):
        return distributions.DiagonalGaussianNonconjNIG

    @property
    def hyperparameter_settings(self):
        return (
                dict(mu_0=np.zeros(2),sigmas_0=np.ones(2),alpha_0=np.ones(2),beta_0=np.ones(2)),
                dict(mu_0=np.zeros(600),sigmas_0=np.ones(600),alpha_0=np.ones(600),beta_0=np.ones(600)),
                )

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu - d2.mu) < 0.25*np.sqrt(d1.mu.shape[0]) \
                and np.linalg.norm(d1.sigmas-d2.sigmas) < 0.5*d1.sigmas.shape[0]

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,d.sigmas))

    @property
    def geweke_nsamples(self):
        return 5000

    @property
    def geweke_data_size(self):
        return 2

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.mu.shape[0])

    ### class-specific

    def test_log_likelihood(self):
        data = np.random.randn(1000,100)

        mu = np.random.randn(100)
        sigmas = np.random.uniform(1,2,size=100)

        d = distributions.DiagonalGaussian(mu=mu,sigmas=sigmas)
        pdf1 = d.log_likelihood(data)

        import scipy.stats as stats
        pdf2 = stats.norm.logpdf(data,loc=mu,scale=np.sqrt(sigmas)).sum(1)

        assert np.allclose(pdf1,pdf2)

    def test_log_likelihood2(self):
        data = np.random.randn(1000,600)

        mu = np.random.randn(600)
        sigmas = np.random.uniform(1,2,size=600)

        d = distributions.DiagonalGaussian(mu=mu,sigmas=sigmas)
        pdf1 = d.log_likelihood(data)

        import scipy.stats as stats
        pdf2 = stats.norm.logpdf(data,loc=mu,scale=np.sqrt(sigmas)).sum(1)

        assert np.allclose(pdf1,pdf2)


@attr('gaussianfixedmean')
class TestGaussianFixedMean(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.GaussianFixedMean

    @property
    def hyperparameter_settings(self):
        return (dict(mu=np.array([1.,2.,3.]),nu_0=5,lmbda_0=np.diag([3.,2.,1.])),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.sigma - d2.sigma) < 0.25

    def geweke_statistics(self,d,data):
        return np.diag(d.sigma)

    @property
    def geweke_nsamples(self):
        return 25000

    @property
    def geweke_data_size(self):
        return 5

    @property
    def geweke_pval(self):
        return 0.05


@attr('gaussianfixedcov')
class TestGaussianFixedCov(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.GaussianFixedCov

    @property
    def hyperparameter_settings(self):
        return (dict(sigma=np.diag([3.,2.,1.]),mu_0=np.array([1.,2.,3.]),sigma_0=np.eye(3)),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1

    def geweke_statistics(self,d,data):
        return d.mu

    @property
    def geweke_nsamples(self):
        return 25000

    @property
    def geweke_data_size(self):
        return 5

    @property
    def geweke_pval(self):
        return 0.05


@attr('gaussiannonconj')
class TestGaussianNonConj(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.GaussianNonConj

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=np.zeros(2),mu_lmbda_0=2*np.eye(2),nu_0=5,sigma_lmbda_0=np.eye(2)),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1 and np.linalg.norm(d1.sigma-d2.sigma) < 0.25

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,np.diag(d.sigma)))

    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 1

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.mu.shape[0])

    @property
    def resample_kwargs(self):
        return dict(niter=10)


@attr('scalargaussiannix')
class TestScalarGaussianNIX(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.ScalarGaussianNIX

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=2.7,kappa_0=2.,sigmasq_0=4.,nu_0=2),)

    def params_close(self,d1,d2):
        return np.abs(d1.mu-d2.mu) < 0.5 and np.abs(d2.sigmasq - d2.sigmasq) < 0.5

    def geweke_statistics(self,d,data):
        return np.array((d.mu,d.sigmasq))

    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 2

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,1)


@attr('scalargaussiannonconjnix')
class TestScalarGaussianNonconjNIX(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.ScalarGaussianNonconjNIX

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=2.7,tausq_0=4.,sigmasq_0=2.,nu_0=2),)

    def params_close(self,d1,d2):
        return np.abs(d1.mu-d2.mu) < 0.1 and np.abs(d2.sigmasq - d2.sigmasq) < 0.25

    def geweke_statistics(self,d,data):
        return np.array((d.mu,d.sigmasq))

    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 2

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,1)
