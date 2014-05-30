from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange, progprint
from pybasicbayes.util.general import sgd_manypass

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

### run SVI

sgdseq = sgd_manypass(tau=0,kappa=0.7,datalist=[data],npasses=50)
for data, rho_t in progprint(sgdseq):
    posteriormodel.meanfield_sgdstep(data, 1., rho_t)

### see if batch mean field can improve on it much

posteriormodel.add_data(data)
scores = posteriormodel.meanfield_coordinate_descent()

print scores

plt.figure()
posteriormodel.plot()

plt.figure()
plt.plot(scores)

plt.show()

