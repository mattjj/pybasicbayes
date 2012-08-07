from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
from matplotlib import cm
import abc

from abstractions import ModelGibbsSampling, ModelMeanField, Distribution
from abstractions import GibbsSampling, MeanField, Collapsed

from distributions import Multinomial
from internals.labels import Labels, CRPLabels

class Mixture(ModelGibbsSampling, ModelMeanField, Distribution):
    '''
    This class is for mixtures of other distributions.
    '''
    def __init__(self,alpha_0,components,weights=None):
        assert len(components) > 0
        self.components = components
        self.weights = Multinomial(alpha_0=alpha_0,K=len(components),weights=weights)

        self.labels_list = []

    def add_data(self,data):
        self.labels_list.append(Labels(data=data,components=self.components,weights=self.weights))

    def generate(self,N,keep=True):
        templabels = Labels(components=self.components,weights=self.weights,N=N) # this samples labels

        out = np.empty(self.components[0].rvs(size=N).shape)
        counts = np.bincount(templabels.z,minlength=len(self.components))
        for idx,(c,count) in enumerate(zip(self.components,counts)):
            out[templabels.z == idx,...] = c.rvs(size=count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

    ### Distribution (so this class can be used as a component in other models)

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

        ### update sweep!
        # ask labels to get weights over z, stored as l.r
        for l in self.labels_list:
            l.meanfieldupdate()

        # pass the weights to pi
        self.weights.meanfieldupdate(None,[l.r for l in self.labels_list]) # None is a placeholder

        # pass the weights to the components
        for idx, c in enumerate(self.components):
            c.meanfieldupdate([l.data for l in self.labels_list],
                    [l.r[:,idx] for l in self.labels_list])

        ### get vlb!
        vlb = 0

        # get labels terms
        vlb += sum(l.get_vlb() for l in self.labels_list)

        # get pi term
        vlb += self.weights.get_vlb()

        # get components terms
        vlb += sum(c.get_vlb() for c in self.components)

        # finally, need the evidence term in the vlb
        for l in self.labels_list:
            vlb += 0.5 * (l.r.sum(0) * np.array([c.expected_log_likelihood(l.data).sum()
                for c in self.components])).sum() # TODO check this sum

        return vlb

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


class CollapsedMixture(ModelGibbsSampling):
    __metaclass__ = abc.ABCMeta

    def _get_counts(self,k):
        return sum(l._get_counts(k) for l in self.labels_list)

    def _get_data_withlabel(self,k):
        return [l._get_data_withlabel(k) for l in self.labels_list]

    def _get_occupied(self):
        return reduce(set.union,(l._get_occupied() for l in self.labels_list),set([]))

    def plot(self):
        plt.figure()
        cmap = cm.get_cmap()
        used_labels = self._get_occupied()
        num_labels = len(used_labels)

        label_colors = {}
        for idx,label in enumerate(used_labels):
            label_colors[label] = idx/(num_labels-1. if num_labels > 1 else 1.)

        for subfigidx,l in enumerate(self.labels_list):
            plt.subplot(len(self.labels_list),1,1+subfigidx)
            # TODO assuming data is 2D
            for label in used_labels:
                if label in l.z:
                    plt.plot(l.data[l.z==label,0],l.data[l.z==label,1],
                            color=cmap(label_colors[label]),ls='None',marker='x')


# TODO profile this
class CRPMixture(CollapsedMixture):
    def __init__(self,alpha_0,obs_distn):
        assert isinstance(obs_distn,Collapsed)
        self.obs_distn = obs_distn
        self.alpha_0 = alpha_0

        self.labels_list = []

    def add_data(self,data):
        self.labels_list.append(CRPLabels(model=self,data=data,alpha_0=self.alpha_0,obs_distn=self.obs_distn))

    def resample_model(self):
        for l in self.labels_list:
            l.resample()

    def generate(self,N,keep=True):
        templabels = CRPLabels(model=self,alpha_0=self.alpha_0,obs_distn=self.obs_distn,N=N)

        counts = np.bincount(templabels.z)
        out = np.empty(self.obs_distn.rvs(size=N).shape)
        for idx, count in enumerate(counts):
            self.obs_distn.resample()
            out[templabels.z == idx,...] = self.obs_distn.rvs(size=count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

