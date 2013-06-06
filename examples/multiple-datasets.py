from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

alpha_0=5.0
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

data1 = priormodel.rvs(400)
data2 = priormodel.rvs(200)

del priormodel

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

posteriormodel.add_data(data1)
posteriormodel.add_data(data2)

for itr in progprint_xrange(200):
    posteriormodel.resample_model()

posteriormodel.plot()
plt.show()
