from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

from .. import distributions as distributions
from mixins import DistributionTester, BigDataGibbsTester, GewekeGibbsTester

@attr('geometric')
class TestGeometric(BigDataGibbsTester,GewekeGibbsTester,DistributionTester):
    @property
    def distribution_class(self):
        return distributions.Geometric

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=2,beta_0=20),dict(alpha_0=5,beta_0=50))

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.p

@attr('poisson')
class TestPoisson(BigDataGibbsTester,GewekeGibbsTester,DistributionTester):
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
class TestNegativeBinomialFixedR(BigDataGibbsTester,GewekeGibbsTester,DistributionTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialFixedR

    @property
    def hyperparameter_settings(self):
        return (dict(r=5,alpha_0=1,beta_0=9),)

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.p

@attr('negbinintr')
class TestNegativeBinomialIntegerR(BigDataGibbsTester,GewekeGibbsTester,DistributionTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerR

    @property
    def hyperparameter_settings(self):
        return (dict(r_discrete_distn=np.r_[0.,0,0,1,1,1],alpha_0=1,beta_0=9),)

    def params_close(self,d1,d2):
        # since it's easy to be off by 1 in r and still look like the same
        # distribution, best just to check moment parameters
        def mean(d):
            return d.r*d.p/(1.-d.p)
        def var(d):
            return mean(d)/(1.-d.p)
        return np.allclose(mean(d1),mean(d2),rtol=0.05) and np.allclose(var(d1),var(d2),rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.005 # since the statistic is on (0,1), it's really sensitive, or something

