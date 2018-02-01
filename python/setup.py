#from distutils.core import setup
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

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
        raise ImportError('The eigency library must be installed before FEW. '
                          'Automatic install with pip failed.')
finally:
    globals()['eigency'] = importlib.import_module('eigency')

package_version = '0.0'

setup(
    name="Fewtwo",
    author='William La Cava',
    author_email='williamlacava@gmail.com',
    url = 'https://lacava.github.io/fewtwo',
    download_url='https://github.com/lacava/fewtwo/releases/tag/'+package_version,
    license='GNU/GPLv3',
    description='Another feature engineering wrapper for ML.',
    zip_safe=True,
    ext_modules = cythonize([Extension(name='fewtwo',
       sources = ["fewtwo.pyx"],                 # our Cython source
       include_dirs = ['../src/','/usr/include/eigen3/']+eigency.get_includes(include_eigen=False),
       extra_compile_args = ['-std=c++0x','-fopenmp'],
       extra_link_args = ['-lshogun'],      
       language='c++'
       )],
       language="c++")
    )
