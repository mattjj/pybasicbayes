from __future__ import division
import numpy as np
import abc, os

from .. import distributions as distributions
from ..util import testing

###################
#  tester mixins  #
###################

class DistributionTester(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def distribution_class(self):
        pass

    @abc.abstractproperty
    def hyperparameter_settings(self):
        pass

class BigDataGibbsTester(DistributionTester):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def params_close(self,distn1,distn2):
        pass


    def big_data_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_big_data, setting_idx, hypparam_dict

    def check_big_data(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(10000)
        d2.resample(data)

        assert self.params_close(d1,d2)

class GewekeGibbsTester(DistributionTester):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def geweke_statistics(self,distn,data):
        pass


    def geweke_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_geweke, setting_idx, hypparam_dict

    def geweke_figure_filepath(self,setting_idx):
        return os.path.join(os.path.dirname(__file__),'figures',
                            self.__class__.__name__,'setting_%d.pdf' % setting_idx)

    def check_geweke(self,setting_idx,hypparam_dict):
        from matplotlib import pyplot as plt
        plt.ioff()
        fig = plt.figure()

        nsamples = 2000
        data_size = 50

        for trial in xrange(3):
            # collect forward-generated statistics
            forward_statistics = []
            for i in xrange(nsamples):
                d = self.distribution_class(**hypparam_dict)
                data = d.rvs(data_size)
                forward_statistics.append(self.geweke_statistics(d,data))

            # collect gibbs-generated statistics
            gibbs_statistics = []
            d = self.distribution_class(**hypparam_dict)
            data = d.rvs(data_size)
            for i in xrange(nsamples):
                d.resample(data)
                data = d.rvs(data_size)
                gibbs_statistics.append(self.geweke_statistics(d,data))

            testing.populations_eq_quantile_plot(forward_statistics,gibbs_statistics,fig=fig)

        import os
        figpath = self.geweke_figure_filepath(setting_idx)
        mkdir(os.path.dirname(figpath))
        plt.savefig(figpath)

############################
#  actual testing classes  #
############################

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


##########
#  misc  #
##########

def mkdir(path):
    # from
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

