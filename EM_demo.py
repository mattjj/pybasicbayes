from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt

from pybasicbayes import models, distributions
from util.text import progprint_xrange

alpha_0=0.5
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

data = priormodel.rvs(100)

del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

fitmodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

fitmodel.add_data(data)

print 'Gibbs Sampling'
for itr in progprint_xrange(50):
    fitmodel.resample_model()

print 'EM'
for itr in progprint_xrange(100):
    fitmodel.EM_step()

print 'BIC: %0.3f' % fitmodel.BIC()

fitmodel.plot()
plt.show()

