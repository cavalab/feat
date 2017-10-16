# Install

Few-two depends on the [Eigen](http://eigen.tuxfamily.org) matrix library for C++ as well as the 
[Shogun](http://shogun.ml) ML library. 

## Ubuntu

It's easy to install the Eigen deb package: 

```
sudo apt-get install libeigen3-dev
```

And the shogun package:
```
sudo add-apt-repository ppa:shogun-toolbox/stable
sudo apt-get update
sudo apt-get install libshogun18
```

Few-two uses cmake to build. So clone the repo, create a build directory, go into it and use cmake 
to generate the make file. Then you can build by calling `make`. 

```
git clone https://github.com/lacava/fewtwo
cd fewtwo
mkdir build
cd build
cmake ..
make
```

