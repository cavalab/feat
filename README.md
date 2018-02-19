# Feat

[![Build Status](https://travis-ci.org/lacava/feat.svg?branch=master)](https://travis-ci.org/lacava/feat)

**Feat** is a feature engineering automation tool that learns new representations of raw data 
to improve classifier and regressor performance. The underlying methods are based on Pareto 
optimization and evolutionary computation to search the space of possible transformations.

Feat wraps around a user-chosen ML method and provides a set of representations that give the best 
performance for that method. Each individual in Feat's population is its own data representation. 

Feat uses the [Shogun C++ ML toolbox](http://shogun.ml) to fit models. 

Check out the [documentation](https://lacava.github.io/feat) for installation and examples. 

## Acknowledgments

This method is being developed to study human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). 

## License

GNU GPLv3
