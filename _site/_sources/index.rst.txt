Feature Engineering Automation Tool
===================================

**FEAT** is a feature engineering automation tool that learns new
representations of raw data to improve classifier and regressor
performance. The underlying methods use Pareto optimization and
evolutionary computation to search the space of possible
transformations.

FEAT wraps around a user-chosen ML method and provides a set of
representations that give the best performance for that method. Each
individual in FEAT’s population is its own data representation.


Contact
-------

Maintained by William La Cava (lacava at childrens dot harvard dot edu)

Acknowledgments
---------------

This work is supported by grant R00-LM012926 from the National Library of Medicine. 
FEAT is being developed to develop predictive health models by the `Cavalab <http://cavalab.org>`__.

Cite
----

1. La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H.. Learning concise representations for regression by evolving networks of trees. ICLR 2019. `arxiv:1807.0091 <https://arxiv.org/abs/1807.00981>`__

2. La Cava, W. & Moore, Jason H. (2020).
Genetic programming approaches to learning fair classifiers.
GECCO 2020.
**Best Paper Award**.
`ACM <https://dl.acm.org/doi/abs/10.1145/3377930.3390157>`__,
`arXiv <https://arxiv.org/abs/2004.13282>`__,
`experiments <https://github.com/lacava/fair_gp>`__

3. La Cava, W., Lee, P.C., Ajmal, I., Ding, X., Cohen, J.B., Solanki, P., Moore, J.H., and Herman, D.S (2021).
Application of concise machine learning to construct accurate and interpretable EHR computable phenotypes.
In Review.
`medRxiv <https://www.medrxiv.org/content/10.1101/2020.12.12.20248005v2>`__,
`experiments <https://bitbucket.org/hermanlab/ehr_feat/>`__

Table of Contents
-----------------

.. toctree::
    :caption: Getting Started
    :maxdepth: 2

    install

.. toctree::
    :caption: User Guide
    :maxdepth: 2

    guide/overview
    guide/basics

.. toctree::
    :caption: Examples
    :maxdepth: 1

    examples/command_line
    examples/archive.ipynb
    examples/longitudinal.ipynb

.. toctree::
    :caption: API
    :maxdepth: 1
     
    py_api
    cpp_api

Index
~~~~~
:ref:`genindex`

