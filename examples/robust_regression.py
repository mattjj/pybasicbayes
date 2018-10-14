# Demo of a robust regression model with multivariate-t distributed noise

import numpy as np
import numpy.random as npr
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.distributions import Regression, RobustRegression

D_out = 1
D_in = 2
N = 100

# Make a regression model and simulate data
A = npr.randn(D_out, D_in)
b = npr.randn(D_out)
Sigma = 0.1 * np.eye(D_out)

true_reg = Regression(A=np.column_stack((A, b)), sigma=Sigma, affine=True)
X = npr.randn(N, D_in)
y = true_reg.rvs(x=X, return_xy=False)

# Corrupt a fraction of the data
inds = npr.rand(N) < 0.1
y[inds] = 3 * npr.randn(inds.sum(), D_out)

# Make a test regression and fit it
std_reg = Regression(nu_0=D_out + 2,
                     S_0=np.eye(D_out),
                     M_0=np.zeros((D_out, D_in+1)),
                     K_0=np.eye(D_in+1),
                     affine=True)

robust_reg = RobustRegression(nu_0=D_out+2,
                      S_0=np.eye(D_out),
                      M_0=np.zeros((D_out, D_in+1)),
                      K_0=np.eye(D_in+1),
                      affine=True)

def _collect(r):
    ll = r.log_likelihood((X, y))[~inds].sum()
    err = ((y - r.predict(X))**2).sum(1)
    mse = np.mean(err[~inds])
    return r.A.copy(), ll, mse

def _update(r):
    r.resample([(X,y)])
    return _collect(r)

# Fit the standard regression
smpls = [_collect(std_reg)]
for _ in progprint_xrange(100):
    smpls.append(_update(std_reg))
smpls = zip(*smpls)
std_As, std_lls, std_mses = tuple(map(np.array, smpls))

# Fit the robust regression
smpls = [_collect(robust_reg)]
for _ in progprint_xrange(100):
    smpls.append(_update(robust_reg))
smpls = zip(*smpls)
robust_As, robust_lls, robust_mses = tuple(map(np.array, smpls))


# Plot the inferred regression function
plt.figure(figsize=(8, 4))
xlim = (-3, 3)
ylim = abs(y).max()
npts = 50
x1, x2 = np.meshgrid(np.linspace(*xlim, npts), np.linspace(*xlim, npts))

plt.subplot(131)
mu = true_reg.predict(np.column_stack((x1.ravel(), x2.ravel())))
plt.imshow(mu.reshape((npts, npts)),
           cmap="RdBu", vmin=-ylim, vmax=ylim,
           alpha=0.8,
           extent=xlim + tuple(reversed(xlim)))
plt.scatter(X[~inds,0], X[~inds,1], c=y[~inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='gray')
plt.scatter(X[inds,0], X[inds,1], c=y[inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='k', linewidths=1)
plt.xlim(xlim)
plt.ylim(xlim)
plt.title("True")

plt.subplot(132)
mu = std_reg.predict(np.column_stack((x1.ravel(), x2.ravel())))
plt.imshow(mu.reshape((npts, npts)),
           cmap="RdBu", vmin=-ylim, vmax=ylim,
           alpha=0.8,
           extent=xlim + tuple(reversed(xlim)))
plt.scatter(X[~inds,0], X[~inds,1], c=y[~inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='gray')
plt.scatter(X[inds,0], X[inds,1], c=y[inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='k', linewidths=1)
plt.xlim(xlim)
plt.ylim(xlim)
plt.title("Standard Regression")

plt.subplot(133)
mu = robust_reg.predict(np.column_stack((x1.ravel(), x2.ravel())))
plt.imshow(mu.reshape((npts, npts)),
           cmap="RdBu", vmin=-ylim, vmax=ylim,
           alpha=0.8,
           extent=xlim + tuple(reversed(xlim)))
plt.scatter(X[~inds,0], X[~inds,1], c=y[~inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='gray')
plt.scatter(X[inds,0], X[inds,1], c=y[inds, 0], cmap="RdBu", vmin=-ylim, vmax=ylim, edgecolors='k', linewidths=1)
plt.xlim(xlim)
plt.ylim(xlim)
plt.title("Robust Regression")


print("True A:   {}".format(true_reg.A))
print("Std A:    {}".format(std_As.mean(0)))
print("Robust A: {}".format(robust_As.mean(0)))

# Plot the log likelihoods and mean squared errors
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(std_lls)
plt.plot(robust_lls)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")

plt.subplot(122)
plt.plot(std_mses, label="Standard")
plt.plot(robust_mses, label="Robust")
plt.legend(loc="upper right")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")

plt.show()
