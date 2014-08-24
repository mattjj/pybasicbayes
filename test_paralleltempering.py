from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.parallel_tempering import ParallelTempering

###########
#  setup  #
###########

alpha_0=5.0
obs_dim = 2

obs_hypparams=dict(
        mu_0=np.zeros(obs_dim),
        alphas_0=2*np.ones(obs_dim),
        betas_0=np.ones(obs_dim),
        nus_0=0.01*np.ones(obs_dim))

#####################
#  data generation  #
#####################

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(5)])

data, _ = priormodel.generate(500)

plt.figure()
priormodel.plot()
plt.title('true model')

del priormodel

###############
#  inference  #
###############

posteriormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.DiagonalGaussian(**obs_hypparams) for itr in range(10)])

posteriormodel.add_data(data)

# NOTE: an empty temp list just runs normal Gibbs
# pt = ParallelTempering(posteriormodel,[])
# pt.run(100,5)

pt = ParallelTempering(posteriormodel,[1.2,1.5])
pt.run(200,10)
for (T1,T2), count in pt.swapcounts.items():
    print 'temperature pair (%0.2f, %0.2f) swapped %d times' % (T1,T2,count)
    print '(%0.3f%% of the time)' % ((count / pt.itercount) * 100)
    print

plt.figure()
pt.unit_temp_model.plot()


plt.show()

