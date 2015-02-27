from distutils.core import setup

PYBASICBAYES_VERSION = "0.1"

# NOTE: cython and moviepy are optional dependencies

setup(name='pybasicbayes',
      version=PYBASICBAYES_VERSION,
      description="Basic utilities for Bayesian inference",
      author='Matt Johnson',
      author_email='mattjj@csail.mit.edu',
      maintainer='Matt Johnson',
      maintainer_email='mattjj@csail.mit.edu',
      packages=['pybasicbayes',
                  'pybasicbayes.internals',
                  'pybasicbayes.examples',
                  'pybasicbayes.util',
                  'pybasicbayes.testing'],
      platforms='ALL',
      keywords=['bayesian', 'inference'],
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "nose"
          ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
      ]
)

