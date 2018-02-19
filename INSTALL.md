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
git clone https://github.com/lacava/feat
cd feat
mkdir build
cd build
cmake ..
make
```

# Running the tests
If you want to run the tests, you need to install [Google
Test](https://github.com/google/googletest). A useful guide to doing so is available
[here](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/). Then you can use
cmake to build the tests. From the repo root,

```
cd tests
mkdir build
cd build
cmake ..
make 
```
