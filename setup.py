from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

PYBASICBAYES_VERSION = "0.1.2"

# NOTE: cython and moviepy are optional dependencies

ext_modules = cythonize('pybasicbayes/**/*.pyx')
setup(name='pybasicbayes',
      version=PYBASICBAYES_VERSION,
      description="Basic utilities for Bayesian inference",
      author='Matthew James Johnson',
      author_email='mattjj@csail.mit.edu',
      url="http://github.com/mattjj/pybasicbayes",
      maintainer='Matthew James Johnson',
      maintainer_email='mattjj@csail.mit.edu',
      packages=['pybasicbayes',
                  'pybasicbayes.distributions',
                  'pybasicbayes.util',
                  'pybasicbayes.testing'],
      platforms='ALL',
      keywords=['bayesian', 'inference'],
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "nose",
          "future",
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
      ],
      ext_modules=ext_modules,
      include_dirs=[np.get_include()])
