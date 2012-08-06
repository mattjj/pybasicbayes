from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt

import models, distributions

blah = models.Mixture(alpha_0=2,components=[distributions.Gaussian(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.01,nu_0=4) for itr in range(20)])

blah.generate(200) # starts blah at truth, cheating!

blah.plot()
plt.title('initial zs')

vals = [blah.meanfield_coordinate_descent_step() for itr in range(50)]

blah.plot()
plt.title('final awesomeness')

plt.figure()
plt.plot(vals)

plt.show()
