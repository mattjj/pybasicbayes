from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

alpha_0=5.0
obs_hypparams=dict(
        mu_0=np.zeros(2),
        alphas_0=2*np.ones(2),
        betas_0=np.ones(2),
        nus_0=0.1*np.ones(2))

priormodel = models.DiagonalGaussianMixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(30)])

data, _ = priormodel.generate(500)

plt.figure()
priormodel.plot()
plt.title('true model')

del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

posteriormodel = models.DiagonalGaussianMixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(30)])

posteriormodel.add_data(data)

for itr in progprint_xrange(100):
    posteriormodel.resample_model()

posteriormodel.plot()

plt.show()

