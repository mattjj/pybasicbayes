from __future__ import division
import numpy as np
na = np.newaxis

from pybasicbayes.abstractions import ModelGibbsSampling, Model, Distribution

from pybasicbayes.observations import Multinomial
from pybasicbayes.internals.labels import Labels

class Mixture(ModelGibbsSampling, Model, Distribution):
    '''
    This class is for mixtures of observation distributions.
    '''
    def __init__(self,alpha,components,weights=None):
        self.components = components
        self.weights = Multinomial(alpha=alpha,K=len(components),weights=weights)

        self.labels_list = []

    def add_data(self,data):
        self.labels_list.append(Labels(data=data,components=self.components,weights=self.weights))

    def generate(self,N,keep=True):
        templabels = Labels(components=self.components,weights=self.weights,N=N) # this samples labels

        counts = np.bincount(templabels.z,minlength=len(self.components))
        out = np.concatenate([c.rvs(size=n) for c,n in zip(self.components,counts)])
        out = out[np.random.permutation(N)]

        if keep:
            self.states_list.append(templabels)

        return out, templabels.z

    ### Distribution

    def log_likeihood(self,x):
        return self.weights.log_likelihood(np.arange(len(self.components))) + \
                np.concatenate([c.log_likelihood(x) for c in self.components]).T

    def resample(self,data):
        # acts like distribution resampling: doesn't remember data, but does
        # update instantiated parameters

        # temporarily add the passed data
        self.add_data(data) # this does one ``resampling'' step for labels
        # we don't resample other labels that might be in labels_list

        # now resample components
        for idx, c in enumerate(self.components):
            c.resample(data=[l.data[l.z == idx] for l in self.labels_list])

        # and weights
        self.weights.resample([l.z for l in self.labels_list])

        # remove the passed data
        self.labels_list.pop()

    ### Gibbs sampling

    def resample_model(self):
        for l in self.labels_list:
            l.resample()

        for idx, c in enumerate(self.components):
            c.resample(data=[l.data[l.z == idx] for l in self.labels_list])

        self.weights.resample([l.z for l in self.labels_list])

