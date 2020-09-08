Feature Engineering Automation Tool
===================================

|Build Status| |License: GPL v3|

**Feat** is a feature engineering automation tool that learns new
representations of raw data to improve classifier and regressor
performance. The underlying methods use Pareto optimization and
evolutionary computation to search the space of possible
transformations.

Feat wraps around a user-chosen ML method and provides a set of
representations that give the best performance for that method. Each
individual in Feat’s population is its own data representation.

Cite
----

La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H..
Learning concise representations for regression by evolving networks of
trees. ICLR 2019. `arxiv:1807.0091 <https://arxiv.org/abs/1807.00981>`__

Bibtex:

::

   @inproceedings{la_cava_learning_2019,
       series = {{ICLR}},
       title = {Learning concise representations for regression by evolving networks of trees},
       url = {https://arxiv.org/abs/1807.00981},
       language = {en},
       booktitle = {International {Conference} on {Learning} {Representations}},
       author = {La Cava, William and Singh, Tilak Raj and Taggart, James and Suri, Srinivas and Moore, Jason H.},
       year = {2019},
   }

Acknowledgments
---------------

This method is being developed to study human disease in the `Epistasis
Lab at UPenn <http://epistasis.org>`__.

License
-------

GNU GPLv3

.. |Build Status| image:: https://travis-ci.org/lacava/feat.svg?branch=master
   :target: https://travis-ci.org/lacava/feat
.. |License: GPL v3| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://github.com/lacava/feat/blob/master/LICENSE

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

