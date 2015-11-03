from __future__ import division
from builtins import range
from builtins import object
import numpy as np

from nose.plugins.attrib import attr

import pybasicbayes.distributions as distributions


@attr('GammaCompoundDirichlet', 'slow')
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
        from pybasicbayes.testing.mixins import mkdir
        figpath = os.path.join(
            os.path.dirname(__file__),'figures',
            self.__class__.__name__,'weaklimittest.pdf')
        mkdir(os.path.dirname(figpath))
        plt.savefig(figpath)
