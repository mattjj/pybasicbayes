from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt

import models, distributions

blah = models.Mixture(alpha_0=1,
        components=[distributions.Gaussian(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.1,nu_0=4)
            for itr in range(30)])

data = blah.rvs(300)

blah.add_data(data)

blah.plot()
plt.title('initial zs')

vals = [blah.meanfield_coordinate_descent_step() for itr in range(50)]

blah.plot()
plt.title('final awesomeness')

plt.figure()
plt.plot(vals)

plt.show()
