#from distutils.core import setup
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
print ( help('modules') ) #Added for debugging Purpose
# the setup file relies on eigency to import its include paths for the
# extension modules. however eigency isn't known as a dependency until after
# setup is parsed; so we need to check for and install eigency before setup.
import importlib
try:
    importlib.import_module('eigency')
except ImportError:
    try:
        import pip
        pip.main(['install', 'eigency'])
    except ImportError:
        raise ImportError('The eigency library must be installed before feat. '
                          'Automatic install with pip failed.')
finally:
    globals()['eigency'] = importlib.import_module('eigency')

package_version = '0.0'
import os
env_params = os.environ.keys() 
if 'EIGEN3_INCLUDE_DIR' in env_params:
    eigen_dir = os.environ['EIGEN3_INCLUDE_DIR'] 
else:
    eigen_dir = '/usr/include/eigen3/'

shogun_include_dir = '/usr/include/'
shogun_lib = '/usr/lib/'
if 'SHOGUN_DIR' in env_params:
    shogun_include_dir = os.environ['SHOGUN_DIR']
if 'SHOGUN_LIB' in env_params:
    shogun_lib = os.environ['SHOGUN_LIB']


setup(
    name="feat",
    author='William La Cava',
    author_email='williamlacava@gmail.com',
    url = 'https://lacava.github.io/feat',
    download_url='https://github.com/lacava/feat/releases/tag/'+package_version,
    license='GNU/GPLv3',
    description='Another feature engineering wrapper for ML.',
    zip_safe=True,
    install_requires=['Numpy>=1.8.2','SciPy>=0.13.3','scikit-learn','Cython'],
    py_modules=['feat','metrics'],
    ext_modules = cythonize([Extension(name='pyfeat',
       sources = ["pyfeat.pyx"],    # our cython source
       include_dirs = ['../src/',eigen_dir,shogun_include_dir]
                      +eigency.get_includes(include_eigen=False),
       extra_compile_args = ['-std=c++14','-fopenmp','-Wno-sign-compare',
                             '-Wno-reorder'],
       library_dirs = [shogun_lib],
       extra_link_args = ['-lshogun'],      
       language='c++'
       )],
       language="c++")
    )
