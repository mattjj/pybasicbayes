from __future__ import division
import numpy as np
na = np.newaxis

from util.stats import sample_discrete_from_log

class Labels(object):
    def __init__(self,components,weights,data=None,N=None,z=None):
        assert data is not None or (N is not None and z is None)

        self.components = components
        self.weights = weights

        if data is None:
            # generating
            self.N = N
            self.generate(N)
        else:
            self.data = data
            self.N = data.shape[0]

            if z is not None:
                self.z = z
            else:
                self.resample()

    def generate(self,N):
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

