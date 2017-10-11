# FEW-two

**Few-two** is a feature engineering wrapper that wraps around the Shogun C++ machine learning 
toolkit and interfaces with scikit-learn. Its purpose is to learn new representations of raw data 
to improve classifier and regressor performance. The underlying methods are based on Pareto 
optimization and evolutionary computation to search the space of possible transformations.

FEW-two is a completely different code base from [Few](https://lacava.github.io/few). The main
differences are:

 - Each individual in FEW-two is its own ML model + data representation, instead of one piece of an
   overall model
 - FEW-two is pure c++
 - FEw-two uses the Shogun C++ ML toolbox instead of Scikit-learn

## Install

Few-two depends on the [Eigen](http://eigen.tuxfamily.org) matrix library for C++ as well as the 
[Shogun](http://shogun.ml) ML library. 

### Ubuntu

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

## Contributing
Please follow the [Github flow](https://guides.github.com/introduction/flow/) guidelines for
contributing to this project. 

## Acknowledgments

This method is being developed to study the genetic causes of human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and 
Data Science](http://warrencenter.upenn.edu).


