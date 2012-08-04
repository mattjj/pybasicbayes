from __future__ import division

from util.stats import sample_discrete_from_log

class Labels(object):
    def __init__(self,components,weights,data=None,N=None,z=None):
        assert data is not None or N is not None

        self.components = components
        self.weights = weights

        if data is None:
            # generating
            self.N = N
            self.generate()
        else:
            self.data = data
            self.N = data.shape[0]

            if z is not None:
                self.z = z
            else:
                self.resample()

    def generate(self):
        raise NotImplementedError

    def resample(self):
        data = self.data
        raise NotImplementedError

    

# TODO but then... how do i use a mixture in the context of another model? well,
# i can treat a model like a distribution too... basically call log_likelihood



# class Mixture(Distribution, GibbsSampling):
#     '''
#     This class is for mixtures of other observation distributions,
#     reusing the multinomial class.
#     member data are:
#         1. alpha: prior parameters for w. if it is a number, equivalent to
#            passing alpha*np.ones(num_components)
#         2. components: a list of distribution objects representing the mixture
#         3. weights: a vector specifying the weight of each mixture component
#     '''
#     def __repr__(self):
#         n_mix = self.n_mix
#         display = 'number of mixture: %s\n' % n_mix
#         display += 'log weight and parameters for mixture k:\n'
#         for k in range(n_mix):
#            display += 'log weight: %s ' % self.weights.log_likelihood(k)
#            display += self.components[k].__repr__() + '\n'
#         return display

#     def __init__(self,alpha,components,weights=None):
#         self.n_mix = n_mix = len(components)

#         alpha = np.array(alpha)
#         assert alpha.ndim == 0 or alpha.ndim == 1
#         if alpha.ndim == 0:
#             alpha = alpha * np.ones(n_mix)
#         else:
#             assert len(alpha) == n_mix

#         self.components = components
#         self.weights = multinomial(alpha)

#         if weights is not None:
#             self.weights.discrete = weights

#     def _log_scores(self,x):
#         '''score for component i on data j is in retval[i,j]'''
#         return self.weights.log_likelihood(np.arange(self.n_mix))[:,na] + \
#                 np.vstack([c.log_likelihood(x) for c in self.components])

#     def resample(self,data=np.array([]),niter=1,**kwargs):
#         n = float(len(data))
#         if n == 0:
#             self.weights.resample()
#             for c in self.components:
#                 c.resample()
#         else:
#             for itr in range(niter):
#                 # sample labels
#                 log_scores = self._log_scores(data)
#                 labels = sample_discrete_from_log(log_scores,axis=0)

#                 # resample weights
#                 self.weights.resample(labels)

#                 # resample component parameters
#                 for idx, c in enumerate(self.components):
#                     c.resample(data[labels == idx])

#     def log_likelihood(self,x):
#         return np.logaddexp.reduce(self._log_scores(x),axis=0)

#     def rvs(self,size=[]):
#         size = np.array(size,ndmin=1)
#         labels = self.weights.rvs(size=int(np.prod(size)))
#         counts = np.bincount(labels)
#         out = np.concatenate([c.rvs(size=(count,)) for count,c in zip(counts,self.components)],axis=0)
#         out = out[np.random.permutation(len(out))] # maybe this shuffle isn't formally necessary
#         return np.reshape(out,np.concatenate((size,(-1,))))

#     @classmethod
#     def test(cls):
#         foo = cls(alpha=3.,components=[Gaussian(np.zeros(2),np.eye(2),0.02,4) for idx in range(4)])
#         data = foo.rvs(200)

#         bar = cls(alpha=2./8,components=[Gaussian(np.zeros(2),np.eye(2),0.02,4) for idx in range(8)])
#         bar.resample(data,niter=50)

#         plt.plot(data[:,0],data[:,1],'kx')
#         for c,weight in zip(bar.components,bar.weights.discrete):
#             if weight > 0.1:
#                 plt.plot(c.mu[0],c.mu[1],'bo',markersize=10)
#         plt.show()

#     def plot(self,*args,**kwargs):
#         warn('plotting not implemented for %s' % type(self))

