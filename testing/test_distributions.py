from __future__ import division
import numpy as np

from .. import distributions as distributions
from mixins import BigDataGibbsTester, GewekeGibbsTester

class TestGeometric(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Geometric

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=2,beta_0=20),dict(alpha_0=5,beta_0=50))

    ### BigDataGibbsTester

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.05)

    ### GewekeGibbsTester

    def geweke_statistics(self,d,data):
        return np.atleast_1d(d.p)

