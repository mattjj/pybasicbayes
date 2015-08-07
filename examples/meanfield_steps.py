from __future__ import division
from builtins import input
from builtins import range
import numpy as np
from matplotlib import pyplot as plt
plt.interactive(True)

from pybasicbayes import models, distributions

GENERATE_DATA = True

###########################
#  generate or load data  #
###########################

alpha_0=5.0
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])
data, _ = priormodel.generate(100)
del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

input() # pause for effect

###############
#  inference  #
###############

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)])

posteriormodel.add_data(data)

vlbs = []
plt.figure(2,figsize=(8,6))
posteriormodel.plot()
plt.figure(3,figsize=(8,6))
while True:
    if input().lower() == 'break': # pause at each iteration
        break

    vlb = posteriormodel.meanfield_coordinate_descent_step()

    plt.figure(2)
    plt.clf()
    posteriormodel.plot()

    plt.figure(3)
    plt.clf()
    vlbs.append(vlb)
    plt.plot(vlbs)

