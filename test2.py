from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import models, distributions

blah = models.CRPMixture(alpha_0=5,
        obs_distn=distributions.Gaussian(np.zeros(2),np.eye(2),kappa_0=0.1,nu_0=4))

data = blah.generate(200)[0]

blah.plot()
plt.show()
