from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
plt.ion()

from pybasicbayes import models, distributions


###############
#  load data  #
###############

data = np.loadtxt('data.txt')

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

##################
#  set up model  #
##################

npr.seed(0)

alpha_0 = 5.
obs_hypparams = dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

model = models.Mixture(
    alpha_0=alpha_0,
    components=[distributions.Gaussian(**obs_hypparams) for itr in range(30)]
)

model.add_data(data)

##############
#  animate!  #
##############

## movie
# try:
#     from moviepy.video.io.bindings import mplfig_to_npimage
#     from moviepy.editor import VideoClip
# except:
#     print "No moviepy found. Quitting..."
#     import sys
#     sys.exit(1)

# fig = plt.figure()
# model.plot(draw=False)
# plt.axis([-8,5,-2,6])

# def make_frame_mpl(t):
#     if (t // 2) % 2:
#         model.meanfield_coordinate_descent_step()
#     else:
#         model.resample_model()
#     model.plot(update=True,draw=False)
#     return mplfig_to_npimage(fig)

# animation = VideoClip(make_frame_mpl, duration=12)
# animation.write_videofile('gibbs.mp4',fps=50)




import itertools, sys, json
for i in itertools.count():
    model.resample_model()
    if i % 3 == 0:
        print(json.dumps(model.to_json_dict()))
        sys.stdout.flush()

