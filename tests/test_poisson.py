from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, \
    GewekeGibbsTester


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
