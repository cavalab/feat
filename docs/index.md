# Feat

[![Build Status](https://travis-ci.org/lacava/feat.svg?branch=master)](https://travis-ci.org/lacava/feat)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/lacava/feat/blob/master/LICENSE)

**Feat** is a feature engineering automation tool that learns new representations of raw data 
to improve classifier and regressor performance. The underlying methods use Pareto 
optimization and evolutionary computation to search the space of possible transformations.

Feat wraps around a user-chosen ML method and provides a set of representations that give the best 
performance for that method. Each individual in Feat's population is its own data representation. 

Feat uses the [Shogun C++ ML toolbox](http://shogun.ml) to fit models. 

Check out the [documentation](https://lacava.github.io/feat) for installation and examples. 

## Cite

La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H. (2018). Learning concise representations for regression by evolving networks of trees. [arxiv:1807.0091](https://arxiv.org/abs/1807.00981)

Bibtex: 
 

	@article{la_cava_learning_2018,
		title = {Learning concise representations for regression by evolving networks of trees},
		url = {https://arxiv.org/abs/1807.00981},
		language = {en},
		author = {La Cava, William and Singh, Tilak Raj and Taggart, James and Suri, Srinivas and Moore, Jason H.},
		month = jul,
		year = {2018}
	}

## Acknowledgments

This method is being developed to study human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). 

## License

GNU GPLv3
