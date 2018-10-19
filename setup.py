from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from distutils.errors import CompileError
from warnings import warn
import os.path

try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

class build_ext(_build_ext):
    # see http://stackoverflow.com/q/19919905 for explanation
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

    # if extension modules fail to build, keep going anyway
    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to build extension modules')

class sdist(_sdist):
    def run(self):
        try:
            from Cython.Build import cythonize
            cythonize(os.path.join('pybasicbayes','**','*.pyx'))
        except:
            warn('Failed to generate extension files from Cython sources')
        finally:
            _sdist.run(self)

ext_modules=[
    Extension(
        'pybasicbayes.util.cstats', ['pybasicbayes/util/cstats.c'],
        extra_compile_args=['-O3','-w']),
]

if use_cython:
    from Cython.Build import cythonize
    try:
        ext_modules = cythonize(os.path.join('pybasicbayes','**','*.pyx'))
    except:
        warn('Failed to generate extension module code from Cython files')

setup(name='pybasicbayes',
      version='0.2.4',
      description="Basic utilities for Bayesian inference",
      author='Matthew James Johnson',
      author_email='mattjj@csail.mit.edu',
      url="http://github.com/mattjj/pybasicbayes",
      packages=[
          'pybasicbayes', 'pybasicbayes.distributions',
          'pybasicbayes.util', 'pybasicbayes.testing', 'pybasicbayes.models'],
      platforms='ALL',
      keywords=[
          'bayesian', 'inference', 'mcmc', 'variational inference',
          'mean field', 'vb'],
      install_requires=["numpy", "scipy", "matplotlib", "nose", "future"],
      setup_requires=['numpy'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
      ],
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext, 'sdist': sdist})
