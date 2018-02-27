#Getting Started

Feat depends on the [Eigen](http://eigen.tuxfamily.org) matrix library for C++ as well as the [Shogun](http://shogun.ml) ML library. Both come in easy packages that work across platforms. To see our installation, check out the [Travis install file](http://github.com/lacava/feat/blob/master/ci/.travis_install.sh).


#Dependencies

##Eigen 

Eigen is a header only package. We need Eigen 3 or greater. 

###Debian/Ubuntu 

On Debian systems, you can grab the package: 

    sudo apt-get install libeigen3-dev

You can also download the headers and put them somewhere. Then you just have to tell cmake where they are with the environmental variable `EIGEN3_INCLUDE_DIR`. Example:

    # grab Eigen 3.3.4
    wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
    tar xzf 3.3.4.tar.gz 
    mkdir eigen-3.3.4 
    mv eigen-eigen*/* eigen-3.3.4
    # set an environmental variable to tell cmake where Eigen is
    export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"

##Shogun

You don't have to compile Shogun, just download the binaries. [Their install guide is good.](https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INSTALL.md#binaries). We've listed two of the options here.

###Anaconda

A good option for Anaconda users is the Shogun Anaconda package. If you use conda, you can get what you need by 

    conda install -c conda-forge shogun-cpp

If you do this, you need cmake to find Anaconda's library and include directories. Set these two variables:

    export SHOGUN_LIB=/home/travis/miniconda/lib/
    export SHOGUN_DIR=/home/travis/miniconda/include/

###Debian/Ubuntu

You can also get the Shogun packages:

    sudo add-apt-repository ppa:shogun-toolbox/nightly -y
    sudo apt-get update -y
    sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
    sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev



#Installing

Feat uses [cmake](https://cmake.org/) to build. It uses the typical set of instructions:

    
    git clone https://github.com/lacava/feat # clone the repo
    cd feat # enter the directory
    ./configure # this runs "mkdir build; cd build; cmake .. " 
    ./install # this runs "make -C build VERBOSE=1 -j8"
 

#Running the tests 

*This is totally optional!* If you want to run the tests, you need to install [Google
Test](https://github.com/google/googletest). A useful guide to doing so is available
[here](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/). Then you can use
cmake to build the tests. From the repo root,


    ./configure tests   
    make test

