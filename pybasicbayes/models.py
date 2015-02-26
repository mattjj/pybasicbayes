from __future__ import division
import numpy as np
na = np.newaxis
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.special as special
import abc, copy
from warnings import warn

from abstractions import ModelGibbsSampling, ModelMeanField, ModelEM
from abstractions import Distribution, GibbsSampling, MeanField, Collapsed, \
        MeanFieldSVI, MaxLikelihood, ModelParallelTempering
from distributions import Categorical, CategoricalAndConcentration
from internals.labels import Labels, CRPLabels
from util.stats import getdatasize

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
        return np.logaddexp.reduce(vals,axis=1)

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
        import parallel

        parallel.model = self
        parallel.labels_list = self.labels_list

        if len(self.components) > 0:
            params = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_sampled_component_params)(idx)
                            for idx in range(len(self.components)))

        for c, p in zip(self.components,params):
            c.parameters = p

    def _resample_labels_joblib(self,num_procs):
        from joblib import Parallel, delayed
        import parallel

        if len(self.labels_list) > 0:
            parallel.model = self

            raw = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_sampled_labels)(idx)
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

    def meanfield_sgdstep(self,minibatch,minibatchfrac,stepsize,**kwargs):
        minibatch = minibatch if isinstance(minibatch,list) else [minibatch]
        mb_labels_list = []
        for data in minibatch:
            self.add_data(data,z=np.empty(data.shape[0]),**kwargs) # NOTE: dummy
            mb_labels_list.append(self.labels_list.pop())

        for l in mb_labels_list:
            l.meanfieldupdate()

        self._meanfield_sgdstep_parameters(mb_labels_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_parameters(self,mb_labels_list,minibatchfrac,stepsize):
        self._meanfield_sgdstep_components(mb_labels_list,minibatchfrac,stepsize)
        self._meanfield_sgdstep_weights(mb_labels_list,minibatchfrac,stepsize)

    def _meanfield_sgdstep_components(self,mb_labels_list,minibatchfrac,stepsize):
        for idx, c in enumerate(self.components):
            c.meanfield_sgdstep(
                    [l.data for l in mb_labels_list],
                    [l.r[:,idx] for l in mb_labels_list],
                    minibatchfrac,stepsize)

    def _meanfield_sgdstep_weights(self,mb_labels_list,minibatchfrac,stepsize):
        self.weights.meanfield_sgdstep(
                None,[l.r for l in mb_labels_list],
                minibatchfrac,stepsize)

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
                    for i,c in label_colors.iteritems() if i in used_labels],
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

        from util.stats import sample_discrete
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
        lognorm = np.logaddexp.reduce(self.weights._alpha_mf)
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

    def meanfield_sgdstep(self,minibatch,weights,minibatchfrac,stepsize):
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

        self._meanfield_sgdstep_parameters(mb_labels_list,minibatchfrac,stepsize)

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
        return np.logaddexp.reduce(vals,axis=1).sum()
