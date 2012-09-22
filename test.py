from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy
import operator

from pymattutil.text import progprint_xrange
from pybasicbayes import models, distributions

alpha_0=2.
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.1,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(50)])

data = priormodel.rvs(200)

del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(50)])

posteriormodel.add_data(data)

allvals = []
allmodels = []
for superitr in range(5):
    vals = []
    for itr in progprint_xrange(100):
        vals.append(posteriormodel.meanfield_coordinate_descent_step())
    # print '%d iterations' % (itr+1)
    allvals.append(vals)
    allmodels.append(copy.deepcopy(posteriormodel))
    for itr in range(50):
        posteriormodel.resample_model()


plt.figure()
for vals in allvals:
    plt.plot(vals)

models_and_scores = sorted([(i,m,v) for i,(m,v)
    in enumerate(zip(allmodels,allvals))],key=operator.itemgetter(1),reverse=True)

models_and_scores[0][1].plot()

plt.show()
