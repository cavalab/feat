# Examples

## Command line example

Fewtwo can be run from the command-line. All of its options are configurable there. 
After a default build, the fewtwo executable will be in the `build` directory.  
From the repo directory, type `./build/fewtwo -h` to see options.

The first argument to the executable should be the dataset file to learn from. This dataset should
be a comma- or tab-delimited file with columns corresponding to features, one column corresponding
to the target, and rows corresponding to samples. the first row should be a header with the names of
the features and target. *The target must be named as class, target, or label in order to be
interpreted correctly.* See the datasets in the examples folder for guidance. 

### ENC problem

We will run Fewtwo on the [energy
efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency) dataset from UCI, which is
included in `examples/d_enc.txt`. 
To run Fewtwo with a population 1000 for 100 generations using a random seed of 42, type

```
./build/fewtwo examples/d_enc.csv -p 1000 -g 100 -rs 42
```

The default verbosity=1, so you will get a printout of summary statistics each generation. The final
output should look like 

     Generation 100/100 [//////////////////////////////////////////////////]
     Min Loss	Median Loss	Time (s)
     8.946e+01	9.023e+01	1.814e+01
     Representation Pareto Front--------------------------------------
     Rank	Complexity	Loss	Representation
     1	8	9.010e+01	[x_0][x_1][x_2][x_3][x_4][x_5][x_6][x_7]
     1	29	8.985e+01	[(x_6-x_5)][log(x_5)][(x_3/x_7)][(x_7/x_6)][(x_5+x_6)][(x_5-x_7)]
     1	32	8.981e+01	[(x_7-x_5)][(x_6/x_5)][(x_6-x_5)][log(x_3)][(x_7-x_6)][(x_7+x_6)][(x_7/x_3)]
     1	43	8.980e+01	[log(x_7)][log(x_3)][(x_5-x_3)][(x_3/x_5)][(x_6/x_3)][(x_3*x_6)][(x_5/x_5)]
     1	46	8.972e+01	[(x_3-x_5)][(x_3*x_5)][log(x_6)][exp(x_6)][(x_3/x_6)][(x_6/x_7)][(x_6/x_3)][(x_7-x_5)]
     1	129	8.960e+01	[((x_6+x_3)-log(x_3))][((x_7/x_7)-(x_6+x_6))][((x_3+x_5)-(x_7*x_7))][((x_6-x_7)/exp(x_6))][((x_6*x_5)+(x_5/x_3))][log((x_5-x_6))][((x_3-x_6)/(x_7*x_6))][(exp(x_7)*(x_6+x_6))]
     1	158	8.950e+01	[exp(log(x_5))][log((x_7/x_6))][((x_7+x_5)-(x_5/x_7))][((x_6+x_5)/(x_6-x_7))][log((x_6-x_5))][log((x_5-x_7))][((x_6-x_7)/(x_3+x_5))][((x_3+x_3)-exp(x_3))][((x_7+x_6)+exp(x_5))]
     1	486	8.946e+01	[(log(exp(x_3))+log((x_7-x_3)))][(exp(exp(x_5))*log((x_7-x_5)))][(((x_6*x_3)-(x_7+x_5))-(log(x_3)-exp(x_5)))][(exp((x_6*x_3))-exp(exp(x_3)))][((log(x_6)-exp(x_6))-(log(x_5)+(x_3/x_6)))][((log(x_3)-(x_6*x_6))*log((x_3+x_5)))][exp(log((x_5/x_3)))][(((x_6+x_5)-(x_6*x_6))+((x_5-x_3)+(x_6/x_7)))]
     1	823	8.946e+01	[exp(((x_6*x_7)*(x_7/x_7)))][exp((log(x_7)-exp(x_7)))][exp((exp(x_6)*(x_6*x_6)))][(((x_6-x_6)+(x_3-x_5))+((x_7/x_3)-(x_6/x_7)))][exp((log(x_3)*log(x_7)))][(exp((x_6+x_6))*(log(x_7)+(x_5+x_3)))][(log((x_7*x_3))+(log(x_5)-(x_5-x_6)))][(((x_3/x_7)-(x_5-x_5))+((x_5*x_6)/(x_6-x_5)))][exp((log(x_7)*exp(x_7)))][(((x_5/x_3)*(x_5/x_3))/(exp(x_5)-exp(x_6)))]
     2	40	8.990e+01	[(x_5/x_3)][log(x_6)][(x_6+x_6)][(x_5*x_6)][(x_3/x_7)][(x_7+x_5)][exp(x_7)]
     2	41	8.982e+01	[(x_3+x_6)][(x_6-x_7)][(x_3/x_5)][(x_6+x_6)][log(x_6)][(x_7+x_5)][(x_5-x_3)][(x_5/x_6)][(x_7/x_6)]
     2	46	8.972e+01	[(x_5+x_6)][(x_7+x_6)][(x_7+x_6)][(x_5+x_5)][(x_7/x_3)][(x_7-x_7)][(x_3-x_7)][(x_6/x_7)][exp(x_3)][log(x_3)]
     2	168	8.969e+01	[exp((x_7+x_3))][(log(x_7)-(x_3/x_3))][((x_6*x_5)-log(x_7))][((x_5+x_3)/(x_7+x_5))][((x_6-x_6)+(x_6-x_3))][log((x_6*x_3))][log((x_6+x_5))][((x_3*x_3)-(x_7/x_5))][log((x_3/x_6))][log((x_5-x_7))]
     2	183	8.967e+01	[log((x_7/x_3))][log(log(x_3))][((x_5+x_6)+(x_7-x_5))][((x_6/x_6)+(x_6*x_5))][log((x_7*x_3))][exp((x_6-x_6))][((x_3/x_5)-(x_6-x_6))][(log(x_3)+(x_6*x_7))][(exp(x_6)*(x_6*x_3))]
     2	193	8.957e+01	[((x_3+x_3)*(x_6+x_5))][log((x_3*x_7))][((x_6/x_3)-(x_5+x_5))][log((x_5-x_6))][(exp(x_5)+(x_6-x_5))][log((x_7*x_3))][((x_5-x_5)/exp(x_3))][(log(x_6)*(x_7+x_7))][(log(x_3)+log(x_7))][((x_5+x_3)/(x_7/x_5))]
     2	710	8.947e+01	[(((x_5+x_6)*(x_7*x_3))-((x_3+x_5)*(x_6-x_5)))][exp(((x_6-x_7)+(x_6/x_7)))][log(log(exp(x_6)))][log(((x_3/x_3)-(x_6/x_7)))][(log((x_5+x_5))/(exp(x_7)*(x_5-x_3)))][(log(exp(x_5))/((x_5+x_7)*(x_5-x_3)))][((exp(x_3)*(x_3-x_3))-log(exp(x_7)))][(((x_3-x_3)*exp(x_3))+((x_5*x_5)+(x_6-x_3)))][(log((x_6+x_5))-(exp(x_5)/(x_6*x_3)))][log(((x_3-x_5)*log(x_7)))]
     
     
     finished
     best representation: [log((x_6-x_5))][(log(x_5)/exp(exp(x_5)))][exp(log(x_7))][((x_3-x_7)/log(x_6))]
     score: 89.460238
     done!

### tab-delimited csv files

When using tab-delimited csv files as input, specify `-sep \\t` or `-sep "\t"` at the command line. 
