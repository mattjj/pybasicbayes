from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions
from pybasicbayes.testing.mixins import BigDataGibbsTester, \
    GewekeGibbsTester


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


@attr('categorical_concentration')
class TestCategorical(GewekeGibbsTester):
#class TestCategorical(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.CategoricalAndConcentration

    @property
    def hyperparameter_settings(self):
        return (dict(a_0=5., b_0=5.0, K=5),)

    @property
    def big_data_size(self):
        return 20000

    def params_close(self,d1,d2):
        return np.allclose(d1.weights,d2.weights,atol=0.05) and \
               np.allclose(d1.alpha_0,d2.alpha_0,atol=0.05)

    def geweke_statistics(self,d,data):
        #return np.concatenate((d.weights, [d.alpha_0]))
        return np.array(d.alpha_0)

    @property
    def geweke_nsamples(self):
        return 3000

    @property
    def geweke_data_size(self):
        return 1 # NOTE: more data usually means slower mixing

    @property
    def geweke_ntrials(self):
        return 1
    
    @property
    def geweke_pval(self):
        return 0.05
