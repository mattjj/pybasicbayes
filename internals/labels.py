from __future__ import division
import numpy as np
na = np.newaxis
import numpy.ma as ma

from ..util.stats import sample_discrete_from_log, sample_discrete

class Labels(object):
    def __init__(self,components,weights,data=None,N=None,z=None,
            initialize_from_prior=True):
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
            elif initialize_from_prior:
                self._generate(len(data))
            else:
                self.resample()

    def _generate(self,N):
        self.z = self.weights.rvs(size=N)

    def resample(self,temp=None):
        data = self.data

        scores = np.hstack([c.log_likelihood(data)[:,na] for c in self.components]) \
                + self.weights.log_likelihood(np.arange(len(self.components)))

        if temp is not None:
            scores /= temp

        self.z = sample_discrete_from_log(scores,axis=1)

    ### Mean Field

    def meanfieldupdate(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)

        # update, see Eq. 10.67 in Bishop
        component_scores = np.empty((N,K))

        for idx, c in enumerate(self.components):
            component_scores[:,idx] = c.expected_log_likelihood(data)

        logpitilde = self.weights.expected_log_likelihood(np.arange(len(self.components)))
        logr = logpitilde + component_scores

        self.r = np.exp(logr - logr.max(1)[:,na])
        self.r /= self.r.sum(1)[:,na]

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

        self.expectations -= self.expectations.max(1)[:,na]
        np.exp(self.expectations,out=self.expectations)
        self.expectations /= self.expectations.sum(1)[:,na]


# TODO get counts/stats from model, as in pyhsmm direct assignment sampler
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
        self.z = np.zeros(N,dtype=np.int)
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

