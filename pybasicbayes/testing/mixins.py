from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import abc, os

from nose.plugins.attrib import attr

import pybasicbayes
from pybasicbayes.util import testing
from future.utils import with_metaclass

class DistributionTester(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractproperty
    def distribution_class(self):
        pass

    @abc.abstractproperty
    def hyperparameter_settings(self):
        pass

class BasicTester(DistributionTester):
    @property
    def basic_data_size(self):
        return 1000

    def loglike_lists_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_loglike_lists, setting_idx, hypparam_dict

    def check_loglike_lists(self,setting_idx,hypparam_dict):
        dist = self.distribution_class(**hypparam_dict)
        data = dist.rvs(size=self.basic_data_size)

        l1 = dist.log_likelihood(data).sum()
        l2 = sum(dist.log_likelihood(d) for d in np.array_split(data,self.basic_data_size))

        assert np.isclose(l1,l2)

    def stats_lists_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_stats_lists, setting_idx, hypparam_dict

    def check_stats_lists(self,setting_idx,hypparam_dict):
        dist = self.distribution_class(**hypparam_dict)
        data = dist.rvs(size=self.basic_data_size)

        if hasattr(dist,'_get_statistics'):
            s1 = dist._get_statistics(data)
            s2 = dist._get_statistics([d for d in np.array_split(data,self.basic_data_size)])

            self._check_stats(s1,s2)

    def _check_stats(self,s1,s2):
        if isinstance(s1,np.ndarray):
            if s1.dtype == np.object:
                assert all(np.allclose(t1,t2) for t1, t2 in zip(s1,s2))
            else:
                assert np.allclose(s1,s2)
        elif isinstance(s1,tuple):
            assert all(np.allclose(ss1,ss2) for ss1,ss2 in zip(s1,s2))

    def missing_data_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_missing_data_stats, setting_idx, hypparam_dict

    def check_missing_data_stats(self,setting_idx,hypparam_dict):
        dist = self.distribution_class(**hypparam_dict)
        data = dist.rvs(size=self.basic_data_size)

        if isinstance(data,np.ndarray):
            data[np.random.randint(2,size=data.shape[0]) == 1] = np.nan

            s1 = dist._get_statistics(data)
            s2 = dist._get_statistics(data[~np.isnan(data).any(1)])

            self._check_stats(s1,s2)

class BigDataGibbsTester(with_metaclass(abc.ABCMeta, DistributionTester)):
    @abc.abstractmethod
    def params_close(self,distn1,distn2):
        pass

    @property
    def big_data_size(self):
        return 20000

    @property
    def big_data_repeats_per_setting(self):
        return 1

    @property
    def big_data_hyperparameter_settings(self):
        return self.hyperparameter_settings

    @attr('random')
    def big_data_Gibbs_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.big_data_hyperparameter_settings):
            for i in range(self.big_data_repeats_per_setting):
                yield self.check_big_data_Gibbs, setting_idx, hypparam_dict

    def check_big_data_Gibbs(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(size=self.big_data_size)
        d2.resample(data)

        assert self.params_close(d1,d2)

class MaxLikelihoodTester(with_metaclass(abc.ABCMeta, DistributionTester)):
    @abc.abstractmethod
    def params_close(self,distn1,distn2):
        pass


    @property
    def big_data_size(self):
        return 20000

    @property
    def big_data_repeats_per_setting(self):
        return 1

    @property
    def big_data_hyperparameter_settings(self):
        return self.hyperparameter_settings


    def maxlike_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.big_data_hyperparameter_settings):
            for i in range(self.big_data_repeats_per_setting):
                yield self.check_maxlike, setting_idx, hypparam_dict

    def check_maxlike(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(size=self.big_data_size)
        d2.max_likelihood(data)

        assert self.params_close(d1,d2)

class GewekeGibbsTester(with_metaclass(abc.ABCMeta, DistributionTester)):
    @abc.abstractmethod
    def geweke_statistics(self,distn,data):
        pass


    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 1 # NOTE: more data usually means slower mixing

    @property
    def geweke_ntrials(self):
        return 3

    @property
    def geweke_pval(self):
        return 0.05

    @property
    def geweke_hyperparameter_settings(self):
        return self.hyperparameter_settings

    def geweke_numerical_slice(self,distn,setting_idx):
        return slice(None)

    @property
    def resample_kwargs(self):
        return {}

    @property
    def geweke_resample_kwargs(self):
        return self.resample_kwargs

    @property
    def geweke_num_statistic_fails_to_tolerate(self):
        return 1


    @attr('slow', 'random')
    def geweke_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.geweke_hyperparameter_settings):
            yield self.check_geweke, setting_idx, hypparam_dict

    def geweke_figure_filepath(self,setting_idx):
        return os.path.join(os.path.dirname(__file__),'figures',
                            self.__class__.__name__,'setting_%d.pdf' % setting_idx)

    def check_geweke(self,setting_idx,hypparam_dict):
        import os
        from matplotlib import pyplot as plt
        plt.ioff()
        fig = plt.figure()
        figpath = self.geweke_figure_filepath(setting_idx)
        mkdir(os.path.dirname(figpath))

        nsamples, data_size, ntrials = self.geweke_nsamples, \
                self.geweke_data_size, self.geweke_ntrials

        d = self.distribution_class(**hypparam_dict)
        sample_dim = np.atleast_1d(self.geweke_statistics(d,d.rvs(size=10))).shape[0]

        num_statistic_fails = 0
        for trial in range(ntrials):
            # collect forward-generated statistics
            forward_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
            for i in range(nsamples):
                d = self.distribution_class(**hypparam_dict)
                data = d.rvs(size=data_size)
                forward_statistics[i] = self.geweke_statistics(d,data)

            # collect gibbs-generated statistics
            gibbs_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
            d = self.distribution_class(**hypparam_dict)
            data = d.rvs(size=data_size)
            for i in range(nsamples):
                d.resample(data,**self.geweke_resample_kwargs)
                data = d.rvs(size=data_size)
                gibbs_statistics[i] = self.geweke_statistics(d,data)

            testing.populations_eq_quantile_plot(forward_statistics,gibbs_statistics,fig=fig)
            try:
                sl = self.geweke_numerical_slice(d,setting_idx)
                testing.assert_populations_eq_moments(
                        forward_statistics[...,sl],gibbs_statistics[...,sl],
                        pval=self.geweke_pval)
            except AssertionError:
                datapath = os.path.join(os.path.dirname(__file__),'figures',
                        self.__class__.__name__,'setting_%d_trial_%d.npz' % (setting_idx,trial))
                np.savez(datapath,fwd=forward_statistics,gibbs=gibbs_statistics)
                example_violating_means = forward_statistics.mean(0), gibbs_statistics.mean(0)
                num_statistic_fails += 1

        plt.savefig(figpath)

        assert num_statistic_fails <= self.geweke_num_statistic_fails_to_tolerate, \
                'Geweke MAY have failed, check FIGURES in %s (e.g. %s vs %s)' \
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

