from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes
import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, MaxLikelihoodTester, \
        GewekeGibbsTester, BasicTester

@attr('geometric')
class TestGeometric(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Geometric

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=2,beta_0=20),dict(alpha_0=5,beta_0=5))

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.5

@attr('poisson')
class TestPoisson(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Poisson

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=30,beta_0=3),)

    def params_close(self,d1,d2):
        return np.allclose(d1.lmbda,d2.lmbda,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.lmbda

@attr('negbinfixedr')
class TestNegativeBinomialFixedR(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialFixedR

    @property
    def hyperparameter_settings(self):
        return (dict(r=5,alpha_0=1,beta_0=9),)

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

@attr('negbinintr')
class TestNegativeBinomialIntegerR(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerR

    @property
    def hyperparameter_settings(self):
        return (dict(r_discrete_distn=np.r_[0.,0,0,1,1,1],alpha_0=5,beta_0=5),)

    def params_close(self,d1,d2):
        # since it's easy to be off by 1 in r and still look like the same
        # distribution, best just to check moment parameters
        def mean(d):
            return d.r*d.p/(1.-d.p)
        def var(d):
            return mean(d)/(1.-d.p)
        return np.allclose(mean(d1),mean(d2),rtol=0.1) and np.allclose(var(d1),var(d2),rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.005 # since the statistic is on (0,1), it's really sensitive, or something

@attr('negbinintr2')
class TestNegativeBinomialIntegerR2(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerR2

    @property
    def hyperparameter_settings(self):
        return (dict(r_discrete_distn=np.r_[0.,0,0,1,1,1],alpha_0=5,beta_0=5),)

    def params_close(self,d1,d2):
        # since it's easy to be off by 1 in r and still look like the same
        # distribution, best just to check moment parameters
        def mean(d):
            return d.r*d.p/(1.-d.p)
        def var(d):
            return mean(d)/(1.-d.p)
        return np.allclose(mean(d1),mean(d2),rtol=0.1) and np.allclose(var(d1),var(d2),rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.005 # since the statistic is on (0,1), it's really sensitive, or something


@attr('negbinintrvariant')
class TestNegativeBinomialIntegerRVariant(TestNegativeBinomialIntegerR):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerRVariant

@attr('categorical')
class TestCategorical(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Categorical

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=5.,K=5),)

    @property
    def big_data_size(self):
        return 20000

    def params_close(self,d1,d2):
        return np.allclose(d1.weights,d2.weights,atol=0.05)

    def geweke_statistics(self,d,data):
        return d.weights

    @property
    def geweke_pval(self):
        return 0.05

@attr('regression')
class TestRegression(BasicTester,BigDataGibbsTester,MaxLikelihoodTester,GewekeGibbsTester):
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

    def geweke_statistics(self,d,data):
        return np.concatenate((d.A.flatten(),np.diag(d.sigma)))

    def geweke_nuerical_slice(self,d,setting_idx):
        return slice(0,d.A.flatten().shape[0])

    @property
    def geweke_ntrials(self):
        return 1 # because it's slow

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
        return 30000

    @property
    def geweke_data_size(self):
        return 1

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,d,setting_idx):
        return slice(0,d.mu.shape[0])

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
        return np.linalg.norm(d1.mu - d2.mu) < 0.1*np.sqrt(d1.mu.shape[0]) \
                and np.linalg.norm(d1.sigmas-d2.sigmas) < 0.25*d1.sigmas.shape[0]

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
        return (dict(sigma=np.diag([3.,2.,1.]),mu_0=np.array([1.,2.,3.]),lmbda_0=np.eye(3)),)

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

@attr('CRP')
class TestCRP(BigDataGibbsTester):
    @property
    def distribution_class(self):
        return distributions.CRP

    @property
    def hyperparameter_settings(self):
        return (dict(a_0=1.,b_0=1./10),)

    @property
    def big_data_size(self):
        return [50]*200

    def params_close(self,d1,d2):
        return np.abs(d1.concentration - d2.concentration) < 1.0

@attr('GammaCompoundDirichlet')
class TestDirichletCompoundGamma(object):
    def test_weaklimit(self):
        a = distributions.CRP(10,1)
        b = distributions.GammaCompoundDirichlet(1000,10,1)

        a.concentration = b.concentration = 10.

        from matplotlib import pyplot as plt

        plt.figure()
        crp_counts = np.zeros(10)
        gcd_counts = np.zeros(10)
        for itr in range(500):
            crp_rvs = np.sort(a.rvs(25))[::-1][:10]
            crp_counts[:len(crp_rvs)] += crp_rvs
            gcd_counts += np.sort(b.rvs(25))[::-1][:10]

        plt.plot(crp_counts/200,gcd_counts/200,'bx-')
        plt.xlim(0,10)
        plt.ylim(0,10)

        import os
        from mixins import mkdir
        figpath = os.path.join(os.path.dirname(__file__),'figures',
                self.__class__.__name__,'weaklimittest.pdf')
        mkdir(os.path.dirname(figpath))
        plt.savefig(figpath)

