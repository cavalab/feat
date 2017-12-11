from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(
           "fewtwo.pyx",                 # our Cython source
           sources=["../src/fewtwo.h"],  # additional source file(s)
           language="c++",             # generate C++ code
           eigency.get_includes(),
           extra_compile_args = ['-std=c++0x','-omp']
      ))