from __future__ import division
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
plt.ion()


from pybasicbayes import models, distributions

###################
#  generate data  #
###################

np.random.seed(1)

alpha_0=5.0
obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.025,nu_0=5)

priormodel = models.Mixture(alpha_0=alpha_0,
        components=[distributions.Gaussian(**obs_hypparams) for itr in range(10)])

data, _ = priormodel.generate(250)

plt.figure()
priormodel.plot()
plt.title('true model')

del priormodel

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

##################
#  set up model  #
##################

model = models.Mixture(
            alpha_0=alpha_0,
            components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)]
            )

model.add_data(data)

##############
#  animate!  #
##############

try:
    import moviepy
except:
    print "No moviepy found. Quitting..."
    sys.exit(1)
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

fig = plt.figure()
model.plot(draw=False)

def make_frame_mpl(t):
    if (t // 1) % 2:
        model.meanfield_coordinate_descent_step()
    else:
        model.resample_model()
    model.plot(update=True,draw=False)
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_mpl, duration=6)
animation.write_videofile('gibbs.mp4',fps=50)

