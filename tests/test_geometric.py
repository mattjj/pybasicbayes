from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, \
    GewekeGibbsTester


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
