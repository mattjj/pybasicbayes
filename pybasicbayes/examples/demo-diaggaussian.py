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

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(30)])

data, _ = priormodel.generate(500)
data2, _ = priormodel.generate(500)

del priormodel

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(30)])

posteriormodel.add_data(data)
posteriormodel.add_data(data2)

for itr in progprint_xrange(100,perline=5):
    posteriormodel.resample_model(labels_jobs=2)

posteriormodel.plot()

plt.show()

