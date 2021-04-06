#from distutils.core import setup
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import subprocess
from version import get_version

# PACKAGE VERSION #####
# package_version = '0.3.1'
package_version = get_version(write=True)
#######################

# the setup file relies on eigency to import its include paths for the
# extension modules. however eigency isn't known as a dependency until after
# setup is parsed; so we need to check for and install eigency before setup.
import importlib
try:
    importlib.import_module('eigency')
except ImportError:
    try:
        print('trying to install eigency prior to setup..')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
            'eigency'])
        # # from pip._internal import main 
        # import pip
        # if hasattr(pip, 'main'):
        #     # NOTE: Older versions of pip use this command:
        #     pip.main(['install', 'eigency'])
        # else:
        #     # Newer versions of pip moved main to _internal:
        #     pip._internal.main(['install', 'eigency'])
    except Exception as e:
        print(e)
        raise ImportError('The eigency library must be installed before feat. '
                          'Automatic install with pip failed.')
finally:
    globals()['eigency'] = importlib.import_module('eigency')

import os
env_params = os.environ.keys() 
if 'EIGEN3_INCLUDE_DIR' in env_params:
    eigen_dir = os.environ['EIGEN3_INCLUDE_DIR'] 
elif 'CONDA_PREFIX' in env_params:
    eigen_dir = os.environ['CONDA_PREFIX']+'/include/eigen3/'
else:
    eigen_dir = '/usr/include/eigen3/'
print('eigen_dir:',eigen_dir)

if 'SHOGUN_DIR' in env_params:
    shogun_include_dir = os.environ['SHOGUN_DIR']
elif 'CONDA_PREFIX' in env_params:
    shogun_include_dir = os.environ['CONDA_PREFIX']+'/include/'
else:
    shogun_include_dir = '/usr/include/'
print('shogun_include_dir:',shogun_include_dir)

if 'SHOGUN_LIB' in env_params:
    shogun_lib = os.environ['SHOGUN_LIB']
elif 'CONDA_PREFIX' in env_params:
    shogun_lib = os.environ['CONDA_PREFIX']+'/lib/'
else:
    shogun_lib = '/usr/lib/'
print('shogun_lib:',shogun_lib)

# get path to feat shared library for linking
cwd = '/'.join(os.getcwd().split('/')[:-1])
feat_lib = cwd + '/build/'
# feat_lib = cwd + '/profile/'
print('package version:',package_version)

setup(
    name="feat",
    version=package_version,
    author='William La Cava',
    author_email='williamlacava@gmail.com',
    url = 'https://lacava.github.io/feat',
    download_url=('https://github.com/lacava/feat/releases/tag/'
        +package_version),
    license='GNU/GPLv3',
    description='A Feature Engineering Automation Tool',
    zip_safe=True,
    install_requires=['Numpy>=1.8.2','SciPy>=0.13.3','scikit-learn','Cython',
        'pandas'],
    py_modules=['feat','metrics','versionstr'],
    ext_modules = cythonize([Extension(name='pyfeat',
        sources =  ["pyfeat.pyx"],    # our cython source
        include_dirs = ['../build/','../src/',eigen_dir,shogun_include_dir]
                          +eigency.get_includes(include_eigen=False),
        extra_compile_args = ['-std=c++1y','-fopenmp','-Wno-sign-compare',
                                 '-Wno-reorder','-Wno-unused-variable'],
        library_dirs = [shogun_lib,feat_lib],
        runtime_library_dirs = [feat_lib],
        extra_link_args = ['-lshogun','-lfeat_lib'],      
        language='c++'
       )],
       language="c++")
    )
