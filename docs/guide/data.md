Feat handles continuous, categorical and boolean data types, as well as sequential (i.e. longitudinal) data. 
By default, FEAT will attempt to infer these data types automatically from the input data. However,
the user may also specify types in the C++ API. 

## Typical use case
For traditional ML tasks, the user specifies data and trains an estimator like so:

```
Feat est;
est.fit(X,y)
```

What types are `X` and `y`? 

- *Python*: Like sklearn, FEAT expects `X` to be a 2d numpy array, and `y` to be a 1d numpy array. 
- *C++*: FEAT expects `X` to be a 2d Eigen `ArrayXd`, and `y` to be a 1d Eigen `VectorXd`.

## Command line

FEAT can learn from a `.csv` or `.tsv` file. In those cases, the target column **must** be labelled one 
of the following:

 - class
 - target
 - label

To use tab-separated data, specify `-sep \\t` at the command line.

## Longitudinal data

Longitudinal data is handled by passing a file of longitudinal data with an identifier that associates
each entry with a row of `X`. The longitudinal data should have the following format: 

id | date | name | value
-- | ---- | ----- | ----
1  | 365  | BloodPressure | 128

Each measurement has a unique identifier (`id`), an integer `date`, a string `name`,
and a `value`. 

The ids are used to associate rows of data with rows/samples in `X`. To do so, the
user inputs a numpy array of the same length as `X`, where each value corresponds
to the `id` value in the longitudinal data associated with that row of `X`. 

For example, 

```python
zfile = 'longitudinal.csv'
ids = np.array([1, ...]) 
est.fit(X,y,zfile,ids)
```

This means that `id=1` associates all data in `Z` with the first row of `X` and `y`.

See [here](./../examples/longitudinal.md) for an example.

