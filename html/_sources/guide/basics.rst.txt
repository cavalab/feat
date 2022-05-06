Basic Usage
===========

Feat handles continuous, categorical and boolean data types, as well as
sequential (i.e. longitudinal) data. By default, FEAT will attempt to
infer these data types automatically from the input data. The user may
also specify types in the C++ API.

Typical use case
----------------

For traditional ML tasks, the user specifies data and trains an
estimator like so:

**python**

.. code:: python

   from feat import Feat

   #here's some random data
   import numpy as np
   X = np.random.rand(100,10)  
   y = np.random.rand(100)

   est = Feat()
   est.fit(X,y)

Note that, in *python*, as in sklearn, FEAT expects ``X`` to be an
:math:`N \times D` numpy array, with :math:`N` samples and :math:`D`
features. ``y`` should be 1d numpy array of length :math:`N`.

**c++**

.. code:: cpp

   #include "feat.h"
   using FT::Feat;
   #include <Eigen/Dense>

   // feed in some data
   Eigen::MatrixXd X(7,2); 
   X << 0,1,  
        0.47942554,0.87758256,  
        0.84147098,  0.54030231,
        0.99749499,  0.0707372,
        0.90929743, -0.41614684,
        0.59847214, -0.80114362,
        0.14112001,-0.9899925;
   X.transposeInPlace();

   Eigen::VectorXd y(7); 
   y << 3.0,  3.59159876,  3.30384889,  2.20720158,  0.57015434,
            -1.20648656, -2.68773747;
   // train 
   Feat est;
   est.fit(X,y);

In *c++*, FEAT expects ``X`` to be transposed, i.e., a
:math:`D \times N` Eigen ``MatrixXd``. ``y`` should be an Eigen
``VectorXd`` of length :math:`N`.

Command line
------------

FEAT can learn from a ``.csv`` or ``.tsv`` file. In those cases, the
target column **must** be labelled one of the following:

-  class
-  target
-  label

To use tab-separated data, specify ``-sep \\t`` at the command line.

If you want to load your own data in a c++ script, you can use Feat’s
built-in ``load_csv`` function.

.. code:: cpp

   #include "feat.h"
   // variables
   string dataset = "my_data.csv";
   MatrixXd X;
   VectorXd y; 
   vector<string> names;
   vector<char> dtypes;
   bool binary_endpoint=false; // true for binary classification
   char delim = ',';   // assuming comma-separated file
   // load the data
   FT::load_csv(dataset,X,y,names,dtypes,binary_endpoint,delim);
   feat.set_feature_names(names);
   feat.set_dtypes(dtypes);

Now ``X`` and ``y`` will have your data, and ``feat`` will know its
types and the names of the variables.

Longitudinal data
-----------------

Longitudinal data is handled by passing a file of longitudinal data with
an identifier that associates each entry with a row of ``X``. The
longitudinal data should have the following format:

== ==== ============= =====
id date name          value
== ==== ============= =====
1  365  BloodPressure 128
== ==== ============= =====

Each measurement has a unique identifier (``id``), an integer ``date``,
a string ``name``, and a ``value``.

The ids are used to associate rows of data with rows/samples in ``X``.
To do so, the user inputs a numpy array of the same length as ``X``,
where each value corresponds to the ``id`` value in the longitudinal
data associated with that row of ``X``.

For example,

.. code:: python

   zfile = 'longitudinal.csv'
   ids = np.array([1, ...]) 
   est.fit(X,y,zfile,ids)

This means that ``id=1`` associates all data in ``Z`` with the first row
of ``X`` and ``y``.

See `here <./../examples/longitudinal.md>`__ for an example.
