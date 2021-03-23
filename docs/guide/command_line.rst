Command line example
====================

| Feat can be run from the command-line. All of its options are
  configurable there. After a default build, the feat executable will be
  in the ``build`` directory.
| From the repo directory, type ``./build/feat -h`` to see options.

The first argument to the executable should be the dataset file to learn
from. This dataset should be a comma- or tab-delimited file with columns
corresponding to features, one column corresponding to the target, and
rows corresponding to samples. the first row should be a header with the
names of the features and target. *The target must be named as class,
target, or label in order to be interpreted correctly.* See the datasets
in the examples folder for guidance.

ENC problem
~~~~~~~~~~~

We will run Feat on the `energy
efficiency <https://archive.ics.uci.edu/ml/datasets/Energy+efficiency>`__
dataset from UCI, which is included in ``docs/examples/data/d_enc.txt``.
To run Feat with a population 1000 for 100 generations using a random
seed of 42, type

::

   ./build/feat docs/examples/d_enc.csv -p 100 -g 100 -r 42

The default verbosity=1, so you will get a printout of summary
statistics each generation. The final output should look like

::

   Generation 100/100 [//////////////////////////////////////////////////]
   Min Loss    Median Loss Median (Max) Size   Time (s)
   2.74477e+00 4.40906e+00 12 (20)         3.91275
   Representation Pareto Front--------------------------------------
   Rank    Complexity  Loss    Representation
   1   1   1.53447e+01     [x4]
   1   2   1.22824e+01     [x6][x4]
   1   3   9.66792e+00     [x2][x6][x4]
   1   4   9.20641e+00     [x0][x3][x4][x6]
   1   5   9.07840e+00     [x0][x1][x3][x4][x6]
   1   6   7.29209e+00     [relu(x2)][x4][x6]
   1   7   7.26300e+00     [relu(x2)][x3][x4][x6]
   1   8   5.92479e+00     [relu(x2)][x6][x0][x1][x3]
   1   9   5.71534e+00     [relu(x2)][x4][x0][x1][x6][x3]
   1   10  5.71534e+00     [relu(x2)][x4][x0][x1][x6][x3][x1]
   1   11  5.71534e+00     [relu(x2)][x4][x0][x1][x6][x3][x0][x1]
   1   12  5.06225e+00     [relu(x2)][x4][x6][(x2*x1)]
   1   13  4.94241e+00     [relu(x2)][x4][x1][x6][(x2*x1)]
   1   14  4.48444e+00     [relu(x2)][x4][x0][x1][x6][(x2*x1)]
   1   15  4.48433e+00     [relu(x2)][x4][x0][x1][x6][x3][(x2*x1)]
   1   16  4.46352e+00     [relu(x2)][x4][x0][x1][x6][(x2*x1)][float(x5)]
   1   18  4.42668e+00     [relu(x2)][x4][x0][x1][x6][sqrt(|x6|)][(x2*x1)]
   1   19  4.39144e+00     [relu(x2)][x4][x0][x1][(x6^3)][(x2*x1)]
   1   20  4.27968e+00     [relu(x2)][x4][x0][x1][x6][(x6^3)][(x2*x1)]
   1   21  4.05328e+00     [relu(x2)][sqrt(|x0|)][x6][x3][x4][(x1*x0)][(x0^2)]
   1   22  3.85010e+00     [relu(x2)][sqrt(|x0|)][x6][x3][(x2/x4)][(x1*x0)]
   1   23  3.22447e+00     [relu(x2)][x0][sqrt(|x0|)][x6][(x2/x4)][x2][(x1*x0)]
   1   24  3.15101e+00     [relu(x2)][x4][x0][sqrt(|x0|)][x6][(x2/x4)][x2][(x1*x0)]
   1   26  3.14630e+00     [relu(x2)][x4][x0][sqrt(|x0|)][x6][(x2/x4)][x2][(x1*x0)][flo...
   1   28  3.12080e+00     [relu(x2)][x4][x0][x6][(x2/x4)][x2][(x1*x0)][log(x0)]
   1   29  3.07627e+00     [relu(x2)][x0][sqrt(|x0|)][x6][(x2/x4)][x2][(x1*x0)][(x6^3)]...
   1   32  3.05042e+00     [relu(x2)][x4][x0][x6][(x2/x4)][x2][(x1*x0)][log(x0)][sqrt(|...
   1   33  3.04806e+00     [relu(x2)][x4][x3][x0][x6][(x2/x4)][x2][(x1*x0)][log(x0)][sq...
   1   38  2.92064e+00     [relu(x2)][x0][sqrt(|x6|)][x6][(x2/x4)][x2][(x1*x0)][log(x0)...
   1   45  2.79345e+00     [relu(x2)][x0][x6][(x2/x4)][x2][(x1*x0)][log(x0)][(x6)^(x3)]...
   1   46  2.78956e+00     [relu(x2)][x0][x3][x6][(x2/x4)][x2][(x1*x0)][log(x0)][(x6)^(...
   1   49  2.74477e+00     [relu(x2)][x0][sqrt(|x6|)][x6][(x2/x4)][x2][(x1*x0)][log(x0)...


   finished
   best training representation: [relu(x2)][x0][sqrt(|x6|)][x6][(x2/x4)][x2][(x1*x0)][log(x0)][(x6)^(x3)][(x6^3)]
   train score: 2.744773
   updating best..
   best validation representation: [relu(x2)][x0][x6][(x2/x4)][x2][(x1*x0)][log(x0)][(x6)^(x3)][(x6^3)]
   validation score: 2.907846
   final_model score: 2.795017

   generating training prediction...
   train score: 2.7950e+00
   train r2: 9.6791e-01
   generating test prediction...
   test score: 2.8997e+00
   test r2: 9.7102e-01
   printing final model
   Feature Weight
   x0  370.995875
   (x1*x0) 202.241059
   x2  159.910905
   relu(x2)    -105.321384
   log(x0) 101.855423
   (x6)^(x3)   85.811398
   (x2/x4) 37.755580
   x6  35.792342
   (x6^3)  25.457528
   done!   
