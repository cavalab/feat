# FEAT

[![Build Status](https://travis-ci.org/lacava/feat.svg?branch=master)](https://travis-ci.org/lacava/feat)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/lacava/feat/blob/master/LICENSE)

**FEAT** is a feature engineering automation tool that learns new representations of raw data 
to improve classifier and regressor performance. The underlying methods use Pareto 
optimization and evolutionary computation to search the space of possible transformations.

FEAT wraps around a user-chosen ML method and provides a set of representations that give the best 
performance for that method. Each individual in FEAT's population is its own data representation. 

FEAT uses the [Shogun C++ ML toolbox](http://shogun.ml) to fit models. 

Check out the [documentation](https://cavalab.org/feat) for installation and examples. 

## References

1. La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H.. Learning concise representations for regression by evolving networks of trees. ICLR 2019. [arxiv:1807.0091](https://arxiv.org/abs/1807.00981)

2. La Cava, W. & Moore, Jason H. (2020).
Genetic programming approaches to learning fair classifiers.
GECCO 2020.
**Best Paper Award**.
[ACM](https://dl.acm.org/doi/abs/10.1145/3377930.3390157),
[arXiv](https://arxiv.org/abs/2004.13282),
[experiments](https://github.com/lacava/fair_gp)

3. La Cava, W., Lee, P.C., Ajmal, I., Ding, X., Cohen, J.B., Solanki, P., Moore, J.H., and Herman, D.S (2021).
Application of concise machine learning to construct accurate and interpretable EHR computable phenotypes.
In Review.
[medRxiv](https://www.medrxiv.org/content/10.1101/2020.12.12.20248005v2),
[experiments](https://bitbucket.org/hermanlab/ehr_feat/)



## Contact

Maintained by William La Cava (william.lacava at childrens.harvard.edu)

## Acknowledgments

This work is supported by grant R00-LM012926 from the National Library of Medicine. 
FEAT is being developed to learn clinical diagnostics in the [Cava Lab at Harvard Medical School](http://cavalab.org). 

## License

GNU GPLv3, see [LICENSE](LICENSE)
