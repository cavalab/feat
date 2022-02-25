#!/usr/bin/env python3

import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.dir_util import remove_tree

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
        print("building extension...")
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        #cfg = "Debug" if self.debug else "Release"
        cfg = "Debug"

        conda_prefix = os.environ['CONDA_PREFIX']

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
            "-DGTEST_INCLUDE_DIRS={}/include/".format(conda_prefix),
            "-DGTEST_LIBRARIES={}/lib/libgtest.so".format(conda_prefix),
            "-DEIGEN3_INCLUDE_DIR={}/include/eigen3/".format(conda_prefix),
            "-Dpybind11_DIR={}/lib/python3.8/site-packages/pybind11/share/cmake/pybind11/".format(conda_prefix),
            "-DPYBIND11_FINDPYTHON=ON",
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
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

# Clean old build/ directory if it exists
try:
    remove_tree("./build")
    print("Removed old build directory.")
except FileNotFoundError:
    print("No existing build directory found - skipping.")

setup(
    name="brushgp",
    version="0.0.1",
    author="William La Cava and Joseph D. Romano",
    author_email="joseph.romano@pennmedicine.upenn.edu",  # can change to Bill
    license="GNU General Public License v3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lacava/brush",
    project_urls={
        "Bug Tracker": "https://github.com/lacava/brush/issues",
    },
    # package_dir={"": "src"},
    # packages=find_packages(where="src"),
    # cmake_install_dir="src/brush",
    python_requires=">=3.6",
    install_requires=[
        'numpy'
    ],
    test_requires=[
        'pytest'
    ],
    ext_modules=[CMakeExtension("brushgp")],
    cmdclass={"build_ext": CMakeBuild},
    test_suite='nose.collector',
    tests_require=['nose', 'pmlb'],
    zip_safe=False,
)
