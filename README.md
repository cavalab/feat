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

## Cite

La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H.. Learning concise representations for regression by evolving networks of trees. ICLR 2019. [arxiv:1807.0091](https://arxiv.org/abs/1807.00981)

Bibtex: 
 
    @inproceedings{la_cava_learning_2019,
        series = {{ICLR}},
        title = {Learning concise representations for regression by evolving networks of trees},
        url = {https://arxiv.org/abs/1807.00981},
        language = {en},
        booktitle = {International {Conference} on {Learning} {Representations}},
        author = {La Cava, William and Singh, Tilak Raj and Taggart, James and Suri, Srinivas and Moore, Jason H.},
        year = {2019},
    }

## Contact

Maintained by William La Cava (william.lacava at childrens.harvard.edu)

## Acknowledgments

This work is supported by grant K99-LM012926 from the National Library of Medicine. 
FEAT is being developed to study human disease in the [Epistasis Lab
at UPenn](http://epistasis.org). 

## License

GNU GPLv3, see [LICENSE](LICENSE)
