from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
from matplotlib import cm

from abstractions import ModelGibbsSampling, ModelMeanField, Model, Distribution
from abstractions import GibbsSampling, MeanField

from observations import Multinomial
from internals.labels import Labels

class Mixture(ModelGibbsSampling, ModelMeanField, Distribution):
    '''
    This class is for mixtures of observation distributions.
    '''
    def __init__(self,alpha_0,components,weights=None):
        self.components = components
        self.weights = Multinomial(alpha_0=alpha_0,K=len(components),weights=weights)

        self.labels_list = []

    def add_data(self,data):
        self.labels_list.append(Labels(data=data,components=self.components,weights=self.weights))

    def generate(self,N,keep=True):
        templabels = Labels(components=self.components,weights=self.weights,N=N) # this samples labels

        counts = np.bincount(templabels.z,minlength=len(self.components))
        out = np.concatenate([c.rvs(size=n) for c,n in zip(self.components,counts)])
        # outpermuted = out[np.random.permutation(N)]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

    ### Distribution

    def log_likelihood(self,x):
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
        assert all(isinstance(c,GibbsSampling) for c in self.components), \
                'Components must implement GibbsSampling'

        for l in self.labels_list:
            l.resample()

        for idx, c in enumerate(self.components):
            c.resample(data=[l.data[l.z == idx] for l in self.labels_list])

        self.weights.resample([l.z for l in self.labels_list])

    ### Mean Field

    def meanfield_coordinate_descent_step(self):
        assert all(isinstance(c,MeanField) for c in self.components), \
                'Components must implement MeanField'
        assert len(self.labels_list) > 0, 'Must have data to run MeanField'

        # ask labels to get weights over z, stored as l.r
        for l in self.labels_list:
            l.compute_responsibilities()

        # pass the weights to pi
        self.weights.meanfieldupdate(None,[l.r for l in self.labels_list])

        # pass the weights to the components
        for idx, c in enumerate(self.components):
            c.meanfieldupdate([l.data for l in self.labels_list],
                    [l.r[:,idx] for l in self.labels_list])

    ### Misc.

    def plot(self,color=None):
        # TODO reduce repeated code between this and hsmm.plot
        plt.figure()
        # for plotting purposes, make sure each l has a z
        # in the mean field case, it will make hard assignments to z, but still
        # use the parameters from the mean field updates
        for l in self.labels_list:
            l.resample()
        cmap = cm.get_cmap()
        label_colors = {}
        used_labels = reduce(set.union,[set(l.z) for l in self.labels_list],set([]))
        num_labels = len(used_labels)
        num_subfig_rows = len(self.labels_list)

        for idx,label in enumerate(used_labels):
            label_colors[label] = idx/(num_labels-1 if num_labels > 1 else 1) if color is None else color

        for subfigidx,l in enumerate(self.labels_list):
            # plot the current observation distributions (and obs. if given)
            plt.subplot(num_subfig_rows,1,1+subfigidx)
            self.components[0]._plot_setup(self.components)
            for label, o in enumerate(self.components):
                if label in l.z:
                    o.plot(color=cmap(label_colors[label]),
                            data=l.data[l.z == label] if l.data is not None else None)

class CollapsedDPMixture(ModelGibbsSampling, Model):
    pass

class DirectAssignmentDPMixture(ModelGibbsSampling, Model):
    pass
