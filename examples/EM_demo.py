from __future__ import division
from builtins import range
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

import pybasicbayes
from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

# EM is really terrible! Here's a demo of how to do it on really easy data

### generate and plot the data

alpha_0=100.
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(6)])

data = priormodel.rvs(200)

del priormodel


plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')


min_num_components, max_num_components = (1,12)
num_tries_each = 5

### search over models using BIC as a model selection criterion

BICs = []
examplemodels = []
for idx, num_components in enumerate(progprint_xrange(min_num_components,max_num_components+1)):
    theseBICs = []
    for i in range(num_tries_each):
        fitmodel = models.Mixture(
                alpha_0=10000, # used for random initialization Gibbs sampling, big means use all components
                components=[distributions.Gaussian(**obs_hypparams) for itr in range(num_components)])

        fitmodel.add_data(data)

        # use Gibbs sampling for initialization
        for itr in range(100):
            fitmodel.resample_model()

        # use EM to fit a model
        for itr in range(50):
            fitmodel.EM_step()

        theseBICs.append(fitmodel.BIC())

    examplemodels.append(copy.deepcopy(fitmodel))
    BICs.append(theseBICs)

plt.figure()
plt.errorbar(
        x=np.arange(min_num_components,max_num_components+1),
        y=[np.mean(x) for x in BICs],
        yerr=[np.std(x) for x in BICs]
        )
plt.xlabel('num components')
plt.ylabel('BIC')

plt.figure()
examplemodels[np.argmin([np.mean(x) for x in BICs])].plot()
plt.title('a decent model')

plt.show()

