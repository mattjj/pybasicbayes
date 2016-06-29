from __future__ import division
from __future__ import absolute_import
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
from functools import reduce
from future.utils import with_metaclass
na = np.newaxis
import scipy.special as special
import abc, copy
from warnings import warn
from scipy.misc import logsumexp

from pybasicbayes.abstractions import ModelGibbsSampling, ModelMeanField, ModelEM
from pybasicbayes.abstractions import Distribution, GibbsSampling, MeanField, Collapsed, \
        MeanFieldSVI, MaxLikelihood, ModelParallelTempering
from pybasicbayes.distributions import Categorical, CategoricalAndConcentration
from pybasicbayes.util.stats import getdatasize, sample_discrete_from_log, sample_discrete


#############################
#  internal labels classes  #
#############################

class Labels(object):
    def __init__(self,model,data=None,N=None,z=None,
            initialize_from_prior=True):
        assert data is not None or (N is not None and z is None)

        self.model = model

        if data is None:
            self._generate(N)
        else:
            self.data = data

            if z is not None:
                self.z = z
            elif initialize_from_prior:
                self._generate(len(data))
            else:
                self.resample()

    def _generate(self,N):
        self.z = self.weights.rvs(N)

    @property
    def N(self):
        return len(self.z)

    @property
    def components(self):
        return self.model.components

    @property
    def weights(self):
        return self.model.weights

    def log_likelihood(self):
        if not hasattr(self,'_normalizer') or self._normalizer is None:
            scores = self._compute_scores()
            self._normalizer = logsumexp(scores[~np.isnan(self.data).any(1)],axis=1).sum()
        return self._normalizer

    def _compute_scores(self):
        data, K = self.data, len(self.components)
        scores = np.empty((data.shape[0],K))
        for idx, c in enumerate(self.components):
            scores[:,idx] = c.log_likelihood(data)
        scores += self.weights.log_likelihood(np.arange(K))
        scores[np.isnan(data).any(1)] = 0. # missing data
        return scores

    def clear_caches(self):
        self._normalizer = None

    ### Gibbs sampling

    def resample(self):
        scores = self._compute_scores()
        self.z, lognorms = sample_discrete_from_log(scores,axis=1,return_lognorms=True)
        self._normalizer = lognorms[~np.isnan(self.data).any(1)].sum()

    def copy_sample(self):
        new = copy.copy(self)
        new.z = self.z.copy()
        return new

    ### Mean Field

    def meanfieldupdate(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N,K))
        for idx, c in enumerate(self.components):
            component_scores[:,idx] = c.expected_log_likelihood(data)
        component_scores = np.nan_to_num(component_scores)

        logpitilde = self.weights.expected_log_likelihood(np.arange(len(self.components)))
        logr = logpitilde + component_scores

        self.r = np.exp(logr - logr.max(1)[:,na])
        self.r /= self.r.sum(1)[:,na]

        # for plotting
        self.z = self.r.argmax(1)

    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        errs = np.seterr(invalid='ignore',divide='ignore')
        prod = self.r*np.log(self.r)
        prod[np.isnan(prod)] = 0. # 0 * -inf = 0.
        np.seterr(**errs)

        logpitilde = self.weights.expected_log_likelihood(np.arange(len(self.components)))

        q_entropy = -prod.sum()
        p_avgengy = (self.r*logpitilde).sum()

        return p_avgengy + q_entropy

    ### EM

    def E_step(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)

        self.expectations = np.empty((N,K))
        for idx, c in enumerate(self.components):
            self.expectations[:,idx] = c.log_likelihood(data)
        self.expectations = np.nan_to_num(self.expectations)

        self.expectations += self.weights.log_likelihood(np.arange(K))

        self.expectations -= self.expectations.max(1)[:,na]
        np.exp(self.expectations,out=self.expectations)
        self.expectations /= self.expectations.sum(1)[:,na]

        self.z = self.expectations.argmax(1)


class CRPLabels(object):
    def __init__(self,model,alpha_0,obs_distn,data=None,N=None):
        assert (data is not None) ^ (N is not None)
        self.alpha_0 = alpha_0
        self.obs_distn = obs_distn
        self.model = model

        if data is None:
            # generating
            self._generate(N)
        else:
            self.data = data
            self._generate(data.shape[0])
            self.resample() # one resampling step

    def _generate(self,N):
        # run a CRP forwards
        alpha_0 = self.alpha_0
        self.z = np.zeros(N,dtype=np.int32)
        for n in range(N):
            self.z[n] = sample_discrete(np.concatenate((np.bincount(self.z[:n]),(alpha_0,))))

    def resample(self):
        al, o = np.log(self.alpha_0), self.obs_distn
        self.z = ma.masked_array(self.z,mask=np.zeros(self.z.shape))
        model = self.model

        for n in np.random.permutation(self.data.shape[0]):
            # mask out n
            self.z.mask[n] = True

            # form the scores and sample them
            ks = list(model._get_occupied())
            scores = np.array([
                np.log(model._get_counts(k))+ o.log_predictive(self.data[n],model._get_data_withlabel(k)) \
                        for k in ks] + [al + o.log_marginal_likelihood(self.data[n])])

            idx = sample_discrete_from_log(scores)
            if idx == scores.shape[0]-1:
                self.z[n] = self._new_label(ks)
            else:
                self.z[n] = ks[idx]

            # sample
            # note: the mask gets fixed by assigning into the array
            self.z[n] = sample_discrete_from_log(np.array(scores))

    def _new_label(self,ks):
        # return a label that isn't already used...
        newlabel = np.random.randint(low=0,high=5*max(ks))
        while newlabel in ks:
            newlabel = np.random.randint(low=0,high=5*max(ks))
        return newlabel


    def _get_counts(self,k):
        return np.sum(self.z == k)

    def _get_data_withlabel(self,k):
        return self.data[self.z == k]

    def _get_occupied(self):
        if ma.is_masked(self.z):
            return set(self.z[~self.z.mask])
        else:
            return set(self.z)


###################
#  model classes  #
###################

class Mixture(ModelGibbsSampling, ModelMeanField, ModelEM, ModelParallelTempering):
    '''
    This class is for mixtures of other distributions.
    '''
    _labels_class = Labels

    def __init__(self,components,alpha_0=None,a_0=None,b_0=None,weights=None,weights_obj=None):
        assert len(components) > 0
        assert (alpha_0 is not None) ^ (a_0 is not None and b_0 is not None) \
                ^ (weights_obj is not None)

        self.components = components

        if alpha_0 is not None:
            self.weights = Categorical(alpha_0=alpha_0,K=len(components),weights=weights)
        elif weights_obj is not None:
            self.weights = weights_obj
        else:
            self.weights = CategoricalAndConcentration(
                    a_0=a_0,b_0=b_0,K=len(components),weights=weights)

        self.labels_list = []

    def add_data(self,data,**kwargs):
        self.labels_list.append(self._labels_class(data=np.asarray(data),model=self,**kwargs))
        return self.labels_list[-1]

    @property
    def N(self):
        return len(self.components)

    def generate(self,N,keep=True):
        templabels = self._labels_class(model=self,N=N)

        out = np.empty(self.components[0].rvs(N).shape)
        counts = np.bincount(templabels.z,minlength=self.N)
        for idx,(c,count) in enumerate(zip(self.components,counts)):
            out[templabels.z == idx,...] = c.rvs(count)

        perm = np.random.permutation(N)
        out = out[perm]
        templabels.z = templabels.z[perm]

        if keep:
            templabels.data = out
            self.labels_list.append(templabels)

        return out, templabels.z

    def _clear_caches(self):
        for l in self.labels_list:
            l.clear_caches()

    def _log_likelihoods(self,x):
        # NOTE: nans propagate as nans
        x = np.asarray(x)
        K = len(self.components)
        vals = np.empty((x.shape[0],K))
        for idx, c in enumerate(self.components):
            vals[:,idx] = c.log_likelihood(x)
        vals += self.weights.log_likelihood(np.arange(K))
        return logsumexp(vals,axis=1)

    def log_likelihood(self,x=None):
        if x is None:
            return sum(l.log_likelihood() for l in self.labels_list)
        else:
            assert isinstance(x,(np.ndarray,list))
            if isinstance(x,list):
                return sum(self.log_likelihood(d) for d in x)
            else:
                self.add_data(x)
                return self.labels_list.pop().log_likelihood()

    ### parallel tempering

    @property
    def temperature(self):
        return self._temperature if hasattr(self,'_temperature') else 1.

    @temperature.setter
    def temperature(self,T):
        self._temperature = T

    @property
    def energy(self):
        energy = 0.
        for l in self.labels_list:
            for label, datum in zip(l.z,l.data):
                energy += self.components[label].energy(datum)
        return energy

    def swap_sample_with(self,other):
        self.components, other.components = other.components, self.components
        self.weights, other.weights = other.weights, self.weights

        for l1, l2 in zip(self.labels_list,other.labels_list):
            l1.z, l2.z = l2.z, l1.z

    ### Gibbs sampling

    def resample_model(self,num_procs=0,components_jobs=0):
        self.resample_components(num_procs=components_jobs)
        self.resample_weights()
        self.resample_labels(num_procs=num_procs)

    def resample_weights(self):
        self.weights.resample([l.z for l in self.labels_list])
        self._clear_caches()

    def resample_components(self,num_procs=0):
        if num_procs == 0:
            for idx, c in enumerate(self.components):
                c.resample(data=[l.data[l.z == idx] for l in self.labels_list])
        else:
            self._resample_components_joblib(num_procs)
        self._clear_caches()

    def resample_labels(self,num_procs=0):
        if num_procs == 0:
            for l in self.labels_list:
                l.resample()
        else:
            self._resample_labels_joblib(num_procs)

    def copy_sample(self):
        new = copy.copy(self)
        new.components = [c.copy_sample() for c in self.components]
        new.weights = self.weights.copy_sample()
        new.labels_list = [l.copy_sample() for l in self.labels_list]
        for l in new.labels_list:
            l.model = new
        return new

    def _resample_components_joblib(self,num_procs):
        from joblib import Parallel, delayed
        from . import parallel_mixture

        parallel_mixture.model = self
        parallel_mixture.labels_list = self.labels_list

        if len(self.components) > 0:
            params = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel_mixture._get_sampled_component_params)(idx)
                            for idx in range(len(self.components)))

        for c, p in zip(self.components,params):
            c.parameters = p

    def _resample_labels_joblib(self,num_procs):
        from joblib import Parallel, delayed
        from . import parallel_mixture

        if len(self.labels_list) > 0:
            parallel_mixture.model = self

            raw = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel_mixture._get_sampled_labels)(idx)
                            for idx in range(len(self.labels_list)))

            for l, (z,normalizer) in zip(self.labels_list,raw):
                l.z, l._normalizer = z, normalizer


    ### Mean Field

    def meanfield_coordinate_descent_step(self):
        assert all(isinstance(c,MeanField) for c in self.components), \
                'Components must implement MeanField'
        assert len(self.labels_list) > 0, 'Must have data to run MeanField'

        self._meanfield_update_sweep()
        return self._vlb()

    def _meanfield_update_sweep(self):
        # NOTE: to interleave mean field steps with Gibbs sampling steps, label
        # updates need to come first, otherwise the sampled updates will be
        # ignored and the model will essentially stay where it was the last time
        # mean field updates were run
        # TODO fix that, seed with sample from variational distribution
        self.meanfield_update_labels()
        self.meanfield_update_parameters()

    def meanfield_update_labels(self):
        for l in self.labels_list:
            l.meanfieldupdate()

    def meanfield_update_parameters(self):
        self.meanfield_update_components()
        self.meanfield_update_weights()

    def meanfield_update_weights(self):
        self.weights.meanfieldupdate(None,[l.r for l in self.labels_list])
        self._clear_caches()

    def meanfield_update_components(self):
        for idx, c in enumerate(self.components):
            c.meanfieldupdate([l.data for l in self.labels_list],
                    [l.r[:,idx] for l in self.labels_list])
        self._clear_caches()

    def _vlb(self):
        vlb = 0.
        vlb += sum(l.get_vlb() for l in self.labels_list)
        vlb += self.weights.get_vlb()
        vlb += sum(c.get_vlb() for c in self.components)
        for l in self.labels_list:
            vlb += np.sum([r.dot(c.expected_log_likelihood(l.data))
                                for c,r in zip(self.components, l.r.T)])

        # add in symmetry factor (if we're actually symmetric)
        if len(set(type(c) for c in self.components)) == 1:
            vlb += special.gammaln(len(self.components)+1)

        return vlb

    ### SVI

    def meanfield_sgdstep(self,minibatch,prob,stepsize,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_labels_list = []
        for data in minibatch:
            self.add_data(data,z=np.empty(data.shape[0]),**kwargs) # NOTE: dummy
            mb_labels_list.append(self.labels_list.pop())

        for l in mb_labels_list:
            l.meanfieldupdate()

        self._meanfield_sgdstep_parameters(mb_labels_list,prob,stepsize)

    def _meanfield_sgdstep_parameters(self,mb_labels_list,prob,stepsize):
        self._meanfield_sgdstep_components(mb_labels_list,prob,stepsize)
        self._meanfield_sgdstep_weights(mb_labels_list,prob,stepsize)

    def _meanfield_sgdstep_components(self,mb_labels_list,prob,stepsize):
        for idx, c in enumerate(self.components):
            c.meanfield_sgdstep(
                    [l.data for l in mb_labels_list],
                    [l.r[:,idx] for l in mb_labels_list],
                    prob,stepsize)

    def _meanfield_sgdstep_weights(self,mb_labels_list,prob,stepsize):
        self.weights.meanfield_sgdstep(
                None,[l.r for l in mb_labels_list],
                prob,stepsize)

    ### EM

    def EM_step(self):
        # assert all(isinstance(c,MaxLikelihood) for c in self.components), \
        #         'Components must implement MaxLikelihood'
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

    @property
    def used_labels(self):
        if len(self.labels_list) > 0:
            label_usages = sum(np.bincount(l.z,minlength=self.N) for l in self.labels_list)
            used_labels, = np.where(label_usages > 0)
        else:
            used_labels = np.argsort(self.weights.weights)[-1:-11:-1]
        return used_labels

    def plot(self,color=None,legend=False,alpha=None,update=False,draw=True):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        artists = []

        ### get colors
        cmap = cm.get_cmap()
        if color is None:
            label_colors = dict((idx,cmap(v))
                for idx, v in enumerate(np.linspace(0,1,self.N,endpoint=True)))
        else:
            label_colors = dict((idx,color) for idx in range(self.N))

        ### plot data scatter
        for l in self.labels_list:
            colorseq = [label_colors[label] for label in l.z]
            if update and hasattr(l,'_data_scatter'):
                l._data_scatter.set_offsets(l.data[:,:2])
                l._data_scatter.set_color(colorseq)
            else:
                l._data_scatter = plt.scatter(l.data[:,0],l.data[:,1],c=colorseq,s=5)
            artists.append(l._data_scatter)

        ### plot parameters
        axis = plt.axis()
        for label, (c, w) in enumerate(zip(self.components,self.weights.weights)):
            artists.extend(
                c.plot(
                    color=label_colors[label],
                    label='%d' % label,
                    alpha=min(0.25,1.-(1.-w)**2)/0.25 if alpha is None else alpha,
                    update=update,draw=False))
        plt.axis(axis)

        ### add legend
        if legend and color is None:
            plt.legend(
                [plt.Rectangle((0,0),1,1,fc=c)
                    for i,c in label_colors.items() if i in used_labels],
                [i for i in label_colors if i in used_labels],
                loc='best', ncol=2)

        if draw: plt.draw()
        return artists


    def to_json_dict(self):
        assert len(self.labels_list) == 1
        data = self.labels_list[0].data
        z = self.labels_list[0].z
        assert data.ndim == 2 and data.shape[1] == 2

        return  {
                    'points':[{'x':x,'y':y,'label':int(label)} for x,y,label in zip(data[:,0],data[:,1],z)],
                    'ellipses':[dict(list(c.to_json_dict().items()) + [('label',i)])
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


class MixtureDistribution(Mixture, GibbsSampling, MeanField, MeanFieldSVI, Distribution):
    '''
    This makes a Mixture act like a Distribution for use in other models
    '''

    def __init__(self,niter=1,**kwargs):
        self.niter = niter
        super(MixtureDistribution,self).__init__(**kwargs)

    @property
    def params(self):
        return dict(weights=self.weights.params,components=[c.params for c in self.components])

    @property
    def hypparams(self):
        return dict(weights=self.weights.hypparams,components=[c.hypparams for c in self.components])

    def energy(self,data):
        # TODO TODO this function is horrible
        assert data.ndim == 1

        if np.isnan(data).any():
            return 0.

        from .util.stats import sample_discrete
        likes = np.array([c.log_likelihood(data) for c in self.components]).reshape((-1,))
        likes += np.log(self.weights.weights)
        label = sample_discrete(np.exp(likes - likes.max()))

        return self.components[label].energy(data)

    def log_likelihood(self,x):
        return self._log_likelihoods(x)

    def resample(self,data):
        # doesn't keep a reference to the data like a model would
        assert isinstance(data,list) or isinstance(data,np.ndarray)

        if getdatasize(data) > 0:
            if not isinstance(data,np.ndarray):
                data = np.concatenate(data)

            self.add_data(data,initialize_from_prior=False)

            for itr in range(self.niter):
                self.resample_model()

            self.labels_list.pop()
        else:
            self.resample_model()

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

    def get_vlb(self):
        from warnings import warn
        warn('Pretty sure this is missing a term, VLB is wrong but updates are fine') # TODO
        vlb = 0.
        # vlb += self._labels_vlb # TODO this part is wrong! we need weights passed in again
        vlb += self.weights.get_vlb()
        vlb += sum(c.get_vlb() for c in self.components)
        return vlb

    def expected_log_likelihood(self,x):
        lognorm = logsumexp(self.weights._alpha_mf)
        return sum(np.exp(a - lognorm) * c.expected_log_likelihood(x)
                for a, c in zip(self.weights._alpha_mf, self.components))

    def meanfieldupdate(self,data,weights,**kwargs):
        # NOTE: difference from parent's method is the inclusion of weights
        if not isinstance(data,(list,tuple)):
            data = [data]
            weights = [weights]
        old_labels = self.labels_list
        self.labels_list = []

        for d in data:
            self.add_data(d,z=np.empty(d.shape[0])) # NOTE: dummy

        self.meanfield_update_labels()
        for l, w in zip(self.labels_list,weights):
            l.r *= w[:,na] # here's where the weights are used
        self.meanfield_update_parameters()

        # self._labels_vlb = sum(l.get_vlb() for l in self.labels_list) # TODO hack

        self.labels_list = old_labels

    def meanfield_sgdstep(self,minibatch,weights,prob,stepsize):
        # NOTE: difference from parent's method is the inclusion of weights
        if not isinstance(minibatch,list):
            minibatch = [minibatch]
            weights = [weights]
        mb_labels_list = []
        for data in minibatch:
            self.add_data(data,z=np.empty(data.shape[0])) # NOTE: dummy
            mb_labels_list.append(self.labels_list.pop())

        for l, w in zip(mb_labels_list,weights):
            l.meanfieldupdate()
            l.r *= w[:,na] # here's where weights are used

        self._meanfield_sgdstep_parameters(mb_labels_list,prob,stepsize)

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


class CollapsedMixture(with_metaclass(abc.ABCMeta, ModelGibbsSampling)):
    def _get_counts(self,k):
        return sum(l._get_counts(k) for l in self.labels_list)

    def _get_data_withlabel(self,k):
        return [l._get_data_withlabel(k) for l in self.labels_list]

    def _get_occupied(self):
        return reduce(set.union,(l._get_occupied() for l in self.labels_list),set([]))

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib import cm
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
    _labels_class = CRPLabels

    def __init__(self,alpha_0,obs_distn):
        assert isinstance(obs_distn,Collapsed)
        self.obs_distn = obs_distn
        self.alpha_0 = alpha_0

        self.labels_list = []

    def add_data(self,data):
        assert len(self.labels_list) == 0
        self.labels_list.append(self._labels_class(model=self,data=np.asarray(data),
            alpha_0=self.alpha_0,obs_distn=self.obs_distn))
        return self.labels_list[-1]

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

        templabels = self._labels_class(model=self,alpha_0=self.alpha_0,obs_distn=self.obs_distn,N=N)

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

    def log_likelihood(self,x, K_extra=1):
        """
        Estimate the log likelihood with samples from
         the model. Draw k_extra components which were not populated by
         the current model in order to create a truncated approximate
         mixture model.
        """
        x = np.asarray(x)
        ks = self._get_occupied()
        K = len(ks)
        K_total = K + K_extra

        # Sample observation distributions given current labels
        obs_distns = []
        for k in range(K):
            o = copy.deepcopy(self.obs_distn)
            o.resample(data=self._get_data_withlabel(k))
            obs_distns.append(o)

        # Sample extra observation distributions from prior
        for k in range(K_extra):
            o = copy.deepcopy(self.obs_distn)
            o.resample()
            obs_distns.append(o)

        # Sample a set of weights
        weights = Categorical(alpha_0=self.alpha_0,
                              K=K_total,
                              weights=None)

        assert len(self.labels_list) == 1
        weights.resample(data=self.labels_list[0].z)

        # Now compute the log likelihood
        vals = np.empty((x.shape[0],K_total))
        for k in range(K_total):
            vals[:,k] = obs_distns[k].log_likelihood(x)

        vals += weights.log_likelihood(np.arange(K_total))
        assert not np.isnan(vals).any()
        return logsumexp(vals,axis=1).sum()
