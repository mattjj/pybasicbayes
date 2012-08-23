from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

from pybasicbayes import models, distributions

alpha_0=0.5
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(50)])

data = priormodel.rvs(300)

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(50)])

posteriormodel.add_data(data)

allvals = []
allmodels = []
for superitr in range(10):
    vals = []
    for itr in range(200):
        vals.append(posteriormodel.meanfield_coordinate_descent_step())
        if len(vals) > 1 and vals[-1] < vals[-2] + 100:
            break
    print '%d iterations' % (itr+1)
    allvals.append(vals)
    allmodels.append(copy.deepcopy(posteriormodel))
    for itr in range(200):
        posteriormodel.resample_model()


plt.figure()
for vals in allvals:
    plt.plot(vals)

bestmodel = np.argmax([lambda v: v[-1] for v in allvals])
allmodels[bestmodel].plot()
plt.title('bestmodel')

worstmodel = np.argmin([lambda v: v[-1] in allvals])
allmodels[worstmodel].plot()
plt.title('worstmodel')

allmodels[0].plot()
plt.title('firstmodel')

plt.show()
