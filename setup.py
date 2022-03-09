#from distutils.core import setup
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.dir_util import remove_tree
from Cython.Build import cythonize
import subprocess
import eigency 

################################################################################
# PACKAGE VERSION #####
# Source: https://github.com/Changaco/version.py

from os.path import dirname, isdir, join
import re


PREFIX = ''

tag_re = re.compile(r'\btag: %s([0-9][^,]*)\b' % PREFIX)
version_re = re.compile('^Version: (.+)$', re.M)


def get_version(write=False):
    # Return the version if it has been injected into the file by git-archive
    try:
        version = tag_re.search('$Format:%D$')
        if version:
            return version.group(1)

        d = dirname(__file__) #+ '/../'

        if isdir(join(d, '.git')):
            # Get the version using "git describe".
            cmd = 'git describe --tags --match %s[0-9]* --dirty' % PREFIX
            try:
                version = subprocess.check_output(
                            cmd.split()).decode().strip()[len(PREFIX):]
            except subprocess.CalledProcessError:
                raise RuntimeError('Unable to get version number from git tags')

            # PEP 440 compatibility
            if '-' in version:
                if version.endswith('-dirty'):
                    raise RuntimeError('The working tree is dirty')
                version = '.post'.join(version.split('-')[:2])

        else:
            # Extract the version from the PKG-INFO file.
            with open(join(d, 'PKG-INFO')) as f:
                version = version_re.search(f.read()).group(1)

        # write version
        if write:
            with open('feat/versionstr.py','w') as wf:
                wf.write('__version__="{}"'.format(version))

        return version
    except Exception as e:
        print('Version error:',e)
        print(
        'Unable to get version number from git tags. Not updating version.')
        with open('feat/versionstr.py','r') as f:
            v =  f.readline().split('=')[-1][1:-1]
            print('returning version=',v)
            return v
        # from versionstr import __version__
        # return __version__
package_version = get_version(write=True)

################################################################################
# set paths
import os

env_params = os.environ.keys() 

if 'CONDA_PREFIX' in env_params:
    ENV_PREFIX = os.environ['CONDA_PREFIX']
    LIB_PATH = os.path.join(ENV_PREFIX,'lib')
    INCLUDE_PATH = os.path.join(ENV_PREFIX,'include')

    EIGEN_DIR = os.environ['CONDA_PREFIX']+'/include/eigen3/'
    SHOGUN_INCLUDE_DIR = INCLUDE_PATH
    SHOGUN_LIB = os.environ['CONDA_PREFIX']+'/lib/'
else:
    # ENV_PREFIX = os.environ
    LIB_PATH = os.environ['LD_LIBRARY_PATH'].split(':')[0]
    INCLUDE_PATH = '/usr/path'


    if 'EIGEN3_INCLUDE_DIR' in env_params:
        EIGEN_DIR = os.environ['EIGEN3_INCLUDE_DIR'] 
    else:
        EIGEN_DIR = '/usr/include/eigen3/'

    if 'SHOGUN_DIR' in env_params:
        SHOGUN_INCLUDE_DIR = os.environ['SHOGUN_DIR']
    else:
        SHOGUN_INCLUDE_DIR = '/usr/include/'


    if 'SHOGUN_LIB' in env_params:
        SHOGUN_LIB = os.environ['SHOGUN_LIB']
    else:
        SHOGUN_LIB = '/usr/lib/'

print('INCLUDE_PATH:',INCLUDE_PATH)
print('LIB_PATH:',LIB_PATH)
print('EIGEN_DIR:',EIGEN_DIR)
print('SHOGUN_INCLUDE_DIR:',SHOGUN_INCLUDE_DIR)
print('SHOGUN_LIB:',SHOGUN_LIB)

################################################################################

################################################################################
# Cmake build extension

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64"
}

with open("README.md", 'r', encoding="utf-8") as fp:
    long_description = fp.read()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)



class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        print("building extension...")
        
        #cfg = "Debug" if self.debug else "Release"
        # cfg = "Debug"
        cfg = "Release"

        # extdir = os.path.abspath(
        #     os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = LIB_PATH

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DSHOGUN_DIR={SHOGUN_INCLUDE_DIR}",
            f"-DSHOGUN_LIB={SHOGUN_LIB}",
            f"-DEIGEN3_INCLUDE_DIR={EIGEN_DIR}",
            f"-DOMP={'OFF' if cfg=='Debug' else 'ON'}",
            f"-DLIB_ONLY=ON" # only build feat library
        ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator:
                cmake_args += ["-GNinja"]

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        os.makedirs(self.build_temp, exist_ok=True)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

# # # Clean old build/ directory if it exists
try:
    remove_tree("./build")
    print("Removed old build directory.")
except FileNotFoundError:
    print("No existing build directory found - skipping.")
################################################################################

print('package version:',package_version)
setup(
    name="feat-ml",
    version=package_version,
    author='William La Cava',
    author_email='williamlacava@gmail.com',
    url = 'https://cavalab.org/feat',
    download_url=('https://github.com/cavalab/feat/releases/tag/'
        +package_version),
    license='GNU/GPLv3',
    description='A Feature Engineering Automation Tool',
    python_requires='>=3',
    install_requires=[
                      'Numpy',
                      'scikit-learn',
                      'Cython',
                      'pandas'
    ],
    # package_dir = {'','feat'},
    packages = ['feat'],
    # py_modules=['feat','metrics','versionstr'],
    ext_modules = ([CMakeExtension("feat.feat_lib")]
                    + cythonize([Extension(name='feat.pyfeat',
                        sources =  ["feat/pyfeat.pyx"],    # our cython source
                        include_dirs = (['src',
                                         EIGEN_DIR, 
                                         SHOGUN_INCLUDE_DIR]
                                   + eigency.get_includes(include_eigen=False)
                                       ),
                        extra_compile_args = ['-std=c++1y',
                                              '-fopenmp',
                                              '-Wno-sign-compare',
                                              '-Wno-reorder',
                                              '-Wno-unused-variable',
                                              '-Wno-deprecated',
                                              '-Wno-deprecated-declarations'
                                             ],
                        library_dirs = [SHOGUN_LIB, LIB_PATH],
                        # runtime_library_dirs = [SHOGUN_LIB,LIB_PATH],
                        extra_link_args = ['-lshogun','-lfeat_lib'],      
                        language='c++'
                       )])
                  ),
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False
)
