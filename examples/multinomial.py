from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

data = np.array(
        [[12,20],
         [12,20],
         [12,20],
         [12,20],
         [12,20],
         [22, 5],
         [22, 5],
         [22, 5],
         [22, 5],
         [22, 5],
         [8, 0],
         [8, 0],
         [8, 0],
         [8, 0],
         [8, 0],
         [1,12],
         [1,12],
         [1,12],
         [1,12],
         [1,12]])

posteriormodel = models.Mixture(alpha_0=2.,
        components=[distributions.Multinomial(K=2,alpha_0=2.) for itr in range(10)])

posteriormodel.add_data(data)

for itr in progprint_xrange(50):
    posteriormodel.resample_model()

