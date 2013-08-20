from __future__ import division
import numpy as np
import abc, os

from nose.plugins.attrib import attr

from ..util import testing

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


    @property
    def big_data_size(self):
        return 20000

    @property
    def big_data_repeats_per_setting(self):
        return 1


    def big_data_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            for i in range(self.big_data_repeats_per_setting):
                yield self.check_big_data, setting_idx, hypparam_dict

    def check_big_data(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(self.big_data_size)
        d2.resample(data)

        assert self.params_close(d1,d2)

class GewekeGibbsTester(DistributionTester):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def geweke_statistics(self,distn,data):
        pass


    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 5 # NOTE: more data usually means slower mixing

    @property
    def geweke_ntrials(self):
        return 3

    @property
    def geweke_pval(self):
        return 0.05

    @property
    def geweke_num_statistic_fails_to_tolerate(self):
        return 1


    @attr('slow')
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

        nsamples, data_size, ntrials = self.geweke_nsamples, \
                self.geweke_data_size, self.geweke_ntrials

        d = self.distribution_class(**hypparam_dict)
        sample_dim = np.atleast_1d(self.geweke_statistics(d,d.rvs(10))).shape[0]

        num_statistic_fails = 0
        for trial in xrange(ntrials):
            # collect forward-generated statistics
            forward_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
            for i in xrange(nsamples):
                d = self.distribution_class(**hypparam_dict)
                data = d.rvs(data_size)
                forward_statistics[i] = self.geweke_statistics(d,data)

            # collect gibbs-generated statistics
            gibbs_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
            d = self.distribution_class(**hypparam_dict)
            data = d.rvs(data_size)
            for i in xrange(nsamples):
                d.resample(data)
                data = d.rvs(data_size)
                gibbs_statistics[i] = self.geweke_statistics(d,data)

            testing.populations_eq_quantile_plot(forward_statistics,gibbs_statistics,fig=fig)
            try:
                testing.assert_populations_eq_moments(forward_statistics,gibbs_statistics,pval=self.geweke_pval)
            except AssertionError:
                example_violating_means = forward_statistics.mean(0), gibbs_statistics.mean(0)
                num_statistic_fails += 1

        import os
        figpath = self.geweke_figure_filepath(setting_idx)
        mkdir(os.path.dirname(figpath))
        plt.savefig(figpath)

        assert num_statistic_fails <= self.geweke_num_statistic_fails_to_tolerate, \
                'Geweke may have failed, check figures in %s (e.g. %s vs %s)' \
                % ((os.path.dirname(figpath),) + example_violating_means)


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

