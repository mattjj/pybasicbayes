from __future__ import division
import numpy as np
na = np.newaxis
import numpy.ma as ma

from util.stats import sample_discrete_from_log, sample_discrete

class Labels(object):
    def __init__(self,components,weights,data=None,N=None,z=None):
        assert data is not None or (N is not None and z is None)

        self.components = components
        self.weights = weights

        if data is None:
            # generating
            self._generate(N)
        else:
            self.data = data

            if z is not None:
                self.z = z
            else:
                self.resample()

    def _generate(self,N):
        self.z = self.weights.rvs(size=N)

    def resample(self):
        data = self.data

        scores = np.hstack([c.log_likelihood(data)[:,na] for c in self.components]) \
                + self.weights.log_likelihood(np.arange(len(self.components)))

        self.z = sample_discrete_from_log(scores,axis=1)

    def compute_responsibilities(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)
        component_scores = np.zeros((N,K))

        for idx, c in enumerate(self.components):
            component_scores[:,idx] = c.expected_log_likelihood(data)

        logr = self.weights.expected_log_likelihood(np.arange(len(self.components))) \
                + component_scores

        self.r = np.exp(logr - logr.max(1)[:,na])
        self.r /= self.r.sum(1)[:,na]

class CRPLabels(object):
    def __init__(self,model,alpha_0,obs_distn,data=None,N=None):
        assert data is not None ^ N is not None
        self.alpha_0 = alpha_0
        self.obs_distn = obs_distn
        self.model = model

        if data is None:
            # generating
            self._generate(N)
        else:
            self.data = data
            self.resample()

    def _generate(self,N):
        # run a CRP forwards
        alpha_0 = self.alpha_0
        self.z = np.zeros(N)
        for n in range(N):
            self.z[n] = sample_discrete(np.concatenate((np.bincount(self.z),alpha_0)))

    def resample(self):
        al, o = np.log(self.alpha_0), self.obs_distn
        self.z = ma.masked_array(self.z)
        model = self.model

        for n in np.random.permutation(self.data.shape[0]):
            # mask out n
            self.z.mask[n] = True

            # form the scores
            scores = [np.log(model._get_counts(k))+o.log_predictive(self.data[n],model._get_data_withlabel(k)) \
                    for k in model._get_occupied()] + [al + o.log_marginal_likelihood(self.data[n])]

            # sample
            # note: the mask gets fixed by assigning into the array
            self.z[n] = sample_discrete_from_log(scores)

    def _get_counts(self,k):
        return np.sum(self.z == k)

    def _get_data_withlabel(self,k):
        return self.data[self.z == k]

    def _get_occupied(self):
        return set(self.z)

