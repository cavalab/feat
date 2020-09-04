Feature Engineering Automation Tool
===================================

**Feat** is a feature engineering automation tool that learns new concise representations of raw data 
for machine learning. 
The underlying methods use Pareto optimization and symbolic regression to search the space of possible transformations.

Feat wraps around a user-chosen ML method and provides a set of representations that give the best performance for that method. 
Each individual in Feat's population is its own data representation. 

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
    examples/longitudinal

.. toctree::
    :caption: API
    :maxdepth: 1
     
    py_api
    `C++ API <api_c.html>`_

Index
~~~~~
:ref:`genindex`

