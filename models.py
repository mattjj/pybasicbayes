from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.special as special
import abc, copy
from warnings import warn

from abstractions import ModelGibbsSampling, ModelMeanField, ModelEM
from abstractions import Distribution, GibbsSampling, MeanField, Collapsed, MaxLikelihood
from distributions import Categorical, CategoricalAndConcentration
from internals.labels import Labels, CRPLabels
from util.stats import getdatasize


class Mixture(ModelGibbsSampling, ModelMeanField, ModelEM):
    '''
    This class is for mixtures of other distributions.
    '''
    def __init__(self,components,alpha_0=None,a_0=None,b_0=None,weights=None):
        assert len(components) > 0
        assert (alpha_0 is not None) ^ (a_0 is not None and b_0 is not None)

        self.components = components

        if alpha_0 is not None:
            self.weights = Categorical(alpha_0=alpha_0,K=len(components),weights=weights)
        else:
            self.weights = CategoricalAndConcentration(
                    a_0=a_0,b_0=b_0,K=len(components),weights=weights)

        self.labels_list = []

    def add_data(self,data,**kwargs):
        self.labels_list.append(Labels(data=np.asarray(data),
            components=self.components,weights=self.weights,
            **kwargs))

    def generate(self,N,keep=True):
        templabels = Labels(components=self.components,weights=self.weights,N=N)

        out = np.empty(self.components[0].rvs(N).shape)
        counts = np.bincount(templabels.z,minlength=len(self.components))
        for idx,(c,count) in enumerate(zip(self.components,counts)):
            out[templabels.z == idx,...] = c.rvs(count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

    def _log_likelihoods(self,x):
        x = np.asarray(x)
        K = len(self.components)
        vals = np.empty((x.shape[0],K))
        for idx, c in enumerate(self.components):
            vals[:,idx] = c.log_likelihood(x)
        vals += self.weights.log_likelihood(np.arange(K))
        assert not np.isnan(vals).any()
        return np.logaddexp.reduce(vals,axis=1)

    def log_likelihood(self,x):
        return self._log_likelihoods(x).sum()

    ### Gibbs sampling

    def resample_model(self,temp=None):
        assert all(isinstance(c,GibbsSampling) for c in self.components), \
                'Components must implement GibbsSampling'
        for idx, c in enumerate(self.components):
            c.resample(data=[l.data[l.z == idx] for l in self.labels_list])

        self.weights.resample([l.z for l in self.labels_list])

        for l in self.labels_list:
            l.resample(temp=temp)

    def copy_sample(self):
        new = copy.copy(self)
        new.components = [c.copy_sample() for c in self.components]
        new.weights = self.weights.copy_sample()
        new.labels_list = [l.copy_sample() for l in self.labels_list]
        return new

    ### Mean Field

    def meanfield_coordinate_descent_step(self):
        assert all(isinstance(c,MeanField) for c in self.components), \
                'Components must implement MeanField'
        assert len(self.labels_list) > 0, 'Must have data to run MeanField'

        # NOTE: to interleave mean field steps with Gibbs sampling steps, label
        # updates needs to come first, otherwise the sampled updates will be
        # ignored and the model will essentially stay where it was the last time
        # mean field updates were run

        ### update sweep!
        # update the label responsibilities
        for l in self.labels_list:
            l.meanfieldupdate()

        # pass the weights to pi
        K = len(self.components)
        self.weights.meanfieldupdate(np.arange(K),[l.r for l in self.labels_list])

        # pass the weights to the components
        for idx, c in enumerate(self.components):
            c.meanfieldupdate([l.data for l in self.labels_list],
                    [l.r[:,idx] for l in self.labels_list])

        ### get vlb!
        vlb = 0.

        # get labels terms
        vlb += sum(l.get_vlb() for l in self.labels_list)

        # get pi term
        vlb += self.weights.get_vlb()

        # get components terms
        vlb += sum(c.get_vlb() for c in self.components)

        # finally, need the evidence term in the vlb
        for l in self.labels_list:
            vlb += np.sum([r.dot(c.expected_log_likelihood(l.data))
                                for c,r in zip(self.components, l.r.T)])

        # add in symmetry factor (if we're actually symmetric)
        if len(set(self.weights.weights)) == 1 and \
                len(set(type(c) for c in self.components)) == 1:
            vlb += special.gammaln(len(self.components)+1)

        return vlb

    ### EM

    def EM_step(self):
        assert all(isinstance(c,MaxLikelihood) for c in self.components), \
                'Components must implement MaxLikelihood'
        assert len(self.labels_list) > 0, 'Must have data to run EM'

        ## E step
        for l in self.labels_list:
            l.E_step()

        ## M step
        # component parameters
        for idx, c in enumerate(self.components):
            c.max_likelihood([l.data for l in self.labels_list],
                    [l.expectations[:,idx] for l in self.labels_list])

        # mixture weights
        self.weights.max_likelihood(np.arange(len(self.components)),
                [l.expectations for l in self.labels_list])

    @property
    def num_parameters(self):
        # NOTE: scikit.learn's gmm.py doesn't count the weights in the number of
        # parameters, but I don't know why they wouldn't. Some convention?
        return sum(c.num_parameters for c in self.components) + self.weights.num_parameters

    def BIC(self,data=None):
        '''
        BIC on the passed data.
        If passed data is None (default), calculates BIC on the model's assigned data.
        '''
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        if data is None:
            assert len(self.labels_list) > 0, \
                    "If not passing in data, the class must already have it. Use the method add_data()"
            return -2*sum(self.log_likelihood(l.data) for l in self.labels_list) + \
                        self.num_parameters * np.log(sum(l.data.shape[0] for l in self.labels_list))
        else:
            return -2*self.log_likelihood(data) + self.num_parameters * np.log(data.shape[0])

    def AIC(self):
        # NOTE: in principle this method computes the AIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert len(self.labels_list) > 0, 'Must have data to get AIC'
        return 2*self.num_parameters - 2*sum(self.log_likelihood(l.data) for l in self.labels_list)

    ### Misc.

    def plot(self,color=None,legend=True):
        cmap = cm.get_cmap()

        if len(self.labels_list) > 0:
            label_colors = {}

            used_labels = reduce(set.union,[set(l.z) for l in self.labels_list],set([]))
            num_labels = len(used_labels)
            num_subfig_rows = len(self.labels_list)

            for idx,label in enumerate(used_labels):
                label_colors[label] = idx/(num_labels-1 if num_labels > 1 else 1) \
                        if color is None else color

            for subfigidx,l in enumerate(self.labels_list):
                # plot the current observation distributions (and obs. if given)
                plt.subplot(num_subfig_rows,1,1+subfigidx)
                for label, o in enumerate(self.components):
                    if label in l.z:
                        o.plot(color=cmap(label_colors[label]),
                                data=(l.data[l.z == label] if l.data is not None else None),
                                label='%d' % label)

            if legend:
                plt.legend(
                        [plt.Rectangle((0,0),1,1,fc=cmap(c))
                            for i,c in label_colors.iteritems() if i in used_labels],
                        [i for i in label_colors if i in used_labels],
                        loc='best'
                        )

        else:
            top10 = np.array(self.components)[np.argsort(self.weights.weights)][-1:-11:-1]
            colors = [cmap(x) for x in np.linspace(0,1,len(top10))] if color is None \
                    else [color]*len(top10)
            for i,(o,c) in enumerate(zip(top10,colors)):
                o.plot(color=c,label='%d' % i)

    def to_json_dict(self):
        assert len(self.labels_list) == 1
        data = self.labels_list[0].data
        z = self.labels_list[0].z
        assert data.ndim == 2 and data.shape[1] == 2

        return  {
                    'points':[{'x':x,'y':y,'label':int(label)} for x,y,label in zip(data[:,0],data[:,1],z)],
                    'ellipses':[dict(c.to_json_dict().items() + [('label',i)])
                        for i,c in enumerate(self.components) if i in z]
                }

    def predictive_likelihoods(self,test_data,forecast_horizons):
        likes = self._log_likelihoods(test_data)
        return [likes[k:] for k in forecast_horizons]

    def block_predictive_likelihoods(self,test_data,blocklens):
        csums = np.cumsum(self._log_likelihoods(test_data))
        outs = []
        for k in blocklens:
            outs.append(csums[k:] - csums[:-k])
        return outs

class MixtureDistribution(Mixture, GibbsSampling, Distribution):
    '''
    This makes a Mixture act like a Distribution for use in other compound models
    '''

    def log_likelihood(self,x):
        return self._log_likelihoods(x)

    def resample(self,data,niter=25,temp=None):
        # doesn't keep a reference to the data like a model would
        assert isinstance(data,list) or isinstance(data,np.ndarray)

        if getdatasize(data) > 0:
            if not isinstance(data,np.ndarray):
                data = np.concatenate(data)

            self.add_data(data)

            for itr in range(niter):
                self.resample_model(temp=temp)

            self.labels_list.pop()
        else:
            self.resample_model(temp=temp)

    def max_likelihood(self,data,weights=None):
        if weights is not None:
            raise NotImplementedError
        assert isinstance(data,list) or isinstance(data,np.ndarray)
        if isinstance(data,list):
            data = np.concatenate(data)

        if getdatasize(data) > 0:
            self.add_data(data)
            self.EM_fit()
            self.labels_list = []

    def plot(self,data=[],color='b',label='',plot_params=True,indices=None):
        # TODO handle indices for 1D
        if not isinstance(data,list):
            data = [data]
        for d in data:
            self.add_data(d)

        for l in self.labels_list:
            l.E_step() # sets l.z to MAP estimates
            for label, o in enumerate(self.components):
                if label in l.z:
                    o.plot(color=color,label=label,
                            data=l.data[l.z == label] if l.data is not None else None)

        for d in data:
            self.labels_list.pop()

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

class CRPMixture(CollapsedMixture):
    def __init__(self,alpha_0,obs_distn):
        assert isinstance(obs_distn,Collapsed)
        self.obs_distn = obs_distn
        self.alpha_0 = alpha_0

        self.labels_list = []

    def add_data(self,data):
        assert len(self.labels_list) == 0
        self.labels_list.append(CRPLabels(model=self,data=np.asarray(data),
            alpha_0=self.alpha_0,obs_distn=self.obs_distn))

    def resample_model(self):
        for l in self.labels_list:
            l.resample()

    def generate(self,N,keep=True):
        warn('not fully implemented')
        # TODO only works if there's no other data in the model; o/w need to add
        # existing data to obs resample. should be an easy update.
        # templabels needs to pay attention to its own counts as well as model
        # counts
        assert len(self.labels_list) == 0

        templabels = CRPLabels(model=self,alpha_0=self.alpha_0,obs_distn=self.obs_distn,N=N)

        counts = np.bincount(templabels.z)
        out = np.empty(self.obs_distn.rvs(N).shape)
        for idx, count in enumerate(counts):
            self.obs_distn.resample()
            out[templabels.z == idx,...] = self.obs_distn.rvs(count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

