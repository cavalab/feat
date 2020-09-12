Installation
============

To see our installation process from scratch, check out the `Travis install file <http://github.com/lacava/feat/blob/master/ci/.travis_install.sh>`_.

Conda
-----

The easiest option for install is to use the `conda environment we provide <http://github.com/lacava/feat/blob/master/ci/test-environment.yml>`_. 
Then the build process is the following:

.. code-block:: bash

    git clone https://github.com/lacava/feat # clone the repo
    cd feat # enter the directory
    conda env create -f ci/test-environment.yml
    conda activate feat-env
    # add some environment variables. 
    export SHOGUN_LIB=$CONDA_PREFIX/lib/
    export SHOGUN_DIR=$CONDA_PREFIX/include/
    export EIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3/
    # install feat
    ./configure # this runs "mkdir build; cd build; cmake .. " 
    ./install # this runs "make -C build VERBOSE=1 -j8; python setup.py install"
    

If you want to roll your own with the dependencies, some other options are shown below. 
In this case, you need to tell the `configure` script where Shogun and Eigen are. 
Edit this lines:

.. code-block:: bash

    export SHOGUN_LIB=/your/shogun/lib/
    export SHOGUN_DIR=/your/shugn/include/
    export EIGEN3_INCLUDE_DIR=/your/eigen/eigen3/


Dependencies
------------

Feat uses `cmake <https://cmake.org/>`_ to build. 
It also depends on the `Eigen <http://eigen.tuxfamily.org>`_ matrix library for C++ as well as the `Shogun <http://shogun.ml>`_ ML library. 
Both come in packages on conda that should work across platforms. 
If you need Eigen and Shogun and don't want to use conda, follow these instructions. 

Eigen 
^^^^^

Eigen is a header only package. We need Eigen 3 or greater. 

**Debian/Ubuntu**

On Debian systems, you can grab the package: 

    sudo apt-get install libeigen3-dev

You can also download the headers and put them somewhere. Then you just have to tell cmake where they are with the environmental variable `EIGEN3_INCLUDE_DIR`. Example:

.. code-block:: bash

    # Eigen 3.3.4
    wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
    tar xzf 3.3.4.tar.gz 
    mkdir eigen-3.3.4 
    mv eigen-eigen*/* eigen-3.3.4
    # set an environmental variable to tell cmake where Eigen is
    export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"

Shogun
^^^^^^

You don't have to compile Shogun, just download the binaries. `Their install guide is good. <https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INSTALL.md#binaries>`_ We've listed two of the options here.


**Debian/Ubuntu**


You can also get the Shogun packages:

.. code-block:: bash

    sudo add-apt-repository ppa:shogun-toolbox/nightly -y
    sudo apt-get update -y
    sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
    sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev

Running the tests 
-----------------

*(optional)* If you want to run the tests, you need to install `Google Test <https://github.com/google/googletest>`_. 
A useful guide to doing so is available `here <https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/>`_. 
Then you can use cmake to build the tests. From the repo root,

.. code-block:: bash

    ./configure tests   # builds the test Makefile
    make -C build tests # compiles the tests
    ./build/tests # runs the tests

