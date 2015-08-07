from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

import pybasicbayes
from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

alpha_0=5.0
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

data, _ = priormodel.generate(500)

plt.figure()
priormodel.plot()
plt.title('true model')

del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

posteriormodel.add_data(data)

allscores = []
allmodels = []
for superitr in range(5):
    # Gibbs sampling to wander around the posterior
    print('Gibbs Sampling')
    for itr in progprint_xrange(100):
        posteriormodel.resample_model()

    # mean field to lock onto a mode
    print('Mean Field')
    scores = [posteriormodel.meanfield_coordinate_descent_step()
                for itr in progprint_xrange(100)]

    allscores.append(scores)
    allmodels.append(copy.deepcopy(posteriormodel))

plt.figure()
for scores in allscores:
    plt.plot(scores)
plt.title('model vlb scores vs iteration')

import operator
models_and_scores = sorted([(m,s[-1]) for m,s
    in zip(allmodels,allscores)],key=operator.itemgetter(1),reverse=True)

plt.figure()
models_and_scores[0][0].plot()
plt.title('best model')

plt.show()
