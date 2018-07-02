# Examples

## Command line example

Feat can be run from the command-line. All of its options are configurable there. 
After a default build, the feat executable will be in the `build` directory.  
From the repo directory, type `./build/feat -h` to see options.

The first argument to the executable should be the dataset file to learn from. This dataset should
be a comma- or tab-delimited file with columns corresponding to features, one column corresponding
to the target, and rows corresponding to samples. the first row should be a header with the names of
the features and target. *The target must be named as class, target, or label in order to be
interpreted correctly.* See the datasets in the examples folder for guidance. 

### ENC problem

We will run Feat on the [energy
efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) dataset from UCI, which is
included in `examples/d_enc.txt`. 
To run Feat with a population 1000 for 100 generations using a random seed of 42, type

```
./build/feat examples/d_enc.csv -p 100 -g 100 -r 42
```

The default verbosity=1, so you will get a printout of summary statistics each generation. The final
output should look like 

    Generation 100/100 [//////////////////////////////////////////////////]
    Min Loss	Median Loss	Median (Max) Size	Time (s)
    2.47920e+00	6.27829e+00	27 (37) 		27.75276
    Representation Pareto Front--------------------------------------
    Rank	Complexity	Loss	Representation
    1	1	1.65439e+01	[x_4]
    1	2	1.25723e+01	[x_0][x_4]
    1	3	1.03673e+01	[x_0][x_4][x_6]
    1	4	1.01868e+01	[x_0][x_3][x_4][x_6]
    1	5	1.00550e+01	[x_0][x_1][x_2][x_4][x_6]
    1	6	9.98319e+00	[x_0][x_1][x_2][x_4][x_6][x_7]
    1	7	9.78112e+00	[(x_1^2)][x_2][x_4][x_6]
    1	8	9.70928e+00	[x_7][(x_1^2)][x_2][x_4][x_6]
    1	9	9.68727e+00	[x_7][(x_1^2)][x_2][x_3][x_4][x_6]
    1	10	9.33169e+00	[x_1][(x_2*x_0)][x_2][x_4][x_6]
    1	11	9.32995e+00	[x_1][(x_2*x_0)][x_2][x_4][x_5][x_6]
    1	12	8.84481e+00	[x_2][(x_1^2)][x_2][x_3][x_4][x_6][(x_3+x_0)]
    1	14	8.08357e+00	[x_6][(x_2/(x_2^2))][x_4]
    1	15	8.08357e+00	[x_6][(x_2/(x_2^2))][x_4][x_4]
    1	17	7.95839e+00	[x_6][(x_2/(x_2^2))][x_4][(x_0-x_4)]
    1	20	7.87072e+00	[x_6][(x_2/(x_2^2))][x_4][(x_6*x_7)]
    1	21	7.55717e+00	[(x_2/(x_2^2))][(x_1+(x_6+x_2))][(x_0-x_4)][x_1]
    1	22	7.47840e+00	[x_6][(x_2/(x_2^2))][x_4][log(x_1)]
    1	23	4.84868e+00	[(x_2/(x_2^2))][(x_1^2)][x_2][x_3][x_4][x_6][(x_3+x_0)]
    1	42	4.25015e+00	[((x_1+x_4)^2)][(x_2/x_1)][(x_6+(x_6+x_2))][(x_1-x_2)][((x_1^2)^2)][(x_0^2)][(x_0*x_2)]
    1	45	4.23304e+00	[((x_1+x_4)^2)][(x_2/x_1)][(x_6+(x_6+x_2))][(x_1-x_2)][((x_1^2)^2)][(x_0^2)][(x_0*x_2)][(x_1-x_5)]
    1	48	3.78240e+00	[((x_1+x_4)^2)][(x_2/x_1)][(x_6+(x_6+x_2))][(x_1-x_2)][((x_1^2)^2)][(x_0^2)][(x_0*x_2)][(x_4+x_3)][(x_1==x_4)]
    1	64	3.02734e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)]
    1	67	2.96073e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)][(x_4-x_7)]
    1	70	2.76775e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)][(x_6*x_7)]
    1	73	2.75930e+00	[x_1][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)][(x_6*x_7)][(x_1-x_2)]
    1	74	2.75432e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)][(x_6*x_7)][(x_6^2)]
    1	78	2.74151e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)][(x_6*x_7)][exp(x_4)]
    1	80	2.57538e+00	[x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_6^2)^2)][((x_1^2)^2)][((x_1+x_4)^2)][(x_6*x_4)][(x_0^2)]
    1	115	2.52790e+00	[((exp(x_3)^2)^2)][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_0^2))][(x_6+x_2)][((x_6^2)^2)][((x_1^2)^2)][((x_1+x_4)^2)][(x_6*x_4)][(x_0^2)]
    1	117	2.49092e+00	[((exp(x_3)^2)^2)][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_0*(x_0^2))][(x_6+(x_1+x_2))][((x_6^2)^2)][((x_1^2)^2)][((x_1+x_4)^2)][(x_6*x_4)][(x_0^2)]
    1	129	2.47920e+00	[((exp(x_3)^2)^2)][(x_2/x_1)][((x_4^2)/((x_2^2)^2))][(x_1*(x_0^2))][(x_6+(x_1+x_2))][((x_6^2)^2)][((x_1^2)^2)][((x_1+x_4)^2)][(x_6*x_4)][(x_0^2)]


    finished
    best training representation: [((exp(x_3)^2)^2)][(x_2/x_1)][((x_4^2)/((x_2^2)^2))][(x_1*(x_0^2))][(x_6+(x_1+x_2))][((x_6^2)^2)][((x_1^2)^2)][((x_1+x_4)^2)][(x_6*x_4)][(x_0^2)]
    train score: 2.479198
    best validation representation: [x_3][(x_2/x_1)][((x_4^2)/(x_2^2))][(x_1*(x_1^2))][(x_6+(x_1+x_2))][((x_1^2)^2)][((x_1+x_4)^2)][(x_0^2)]
    validation score: 3.596077
    final_model score: 3.202949
    generating training prediction...
    predicting with best_ind
    train score: 3.20295e+00
    generating test prediction...
    predicting with best_ind
    test score: 4.24600e+00
    done!


### tab-delimited csv files

When using tab-delimited csv files as input, specify `-sep \\t` or `-sep "\t"` at the command line. 

### Feat Cross Validator

The cross-validation version of Feat named `feat_cv` is also built and present in the build/ directory. For cross validation, there are several hyperparameters of Feat that can be tuned: 

* pop_size
* generations
* ml
* max_stall
* selection
* survival
* cross_rate
* functions
* max_depth
* max_dim
* erc
* objectives
* feedback

There are 2 ways to set the hyper parameters for feat\_cv. The first method is to define a string in `cv_main.cc` file and pass that to the feat\_cv constructor. See `cv_main.cc` for details. 

The second method is to create a input file containing group of parameters and pass the filepath using `-infile` flag. Check featcvinput.txt for a sample input file. The input file contains a string similar to the one in `cv_main.cc`.

The general structure of input file is
`[{'token1': (val1, val2, val3), 'token2': (val1, val2)}, {'token1': (val1, val2, val3), 'token2': (val1, val2)}]`
