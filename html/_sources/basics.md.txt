# Basic Usage

Feat handles continuous, categorical and boolean data types, as well as
sequential (i.e. longitudinal) data. By default, FEAT will attempt to
infer these data types automatically from the input data. The user may
also specify types in the C++ API.

## Typical use case

For traditional ML tasks, the user specifies data and trains an
estimator like so:

**python**

``` python
from feat import Feat

#here's some random data
import numpy as np
X = np.random.rand(100,10)  
y = np.random.rand(100)

est = Feat()
est.fit(X,y)
```

Note that, in *python*, as in sklearn, FEAT expects `X` to be an
$N \times D$ numpy array, with $N$ samples and $D$ features. 
`y` should be 1d numpy array of length $N$.

## Longitudinal data

Feat can also handle longitudinal data, which is passed as an additional parameter named Z:

``` python
est.fit(X,y,Z)
```

See [here](examples/longitudinal) for an example.

FEAT expects longitudinal data in the following format:

```python

Z = {
    'variable': ([patient1_values, patient2_values], [patient1_times, patient2_timestamps])
}
```

Longitudinal data is a dictionary in which the keys are the variable names and the values are tuples. 
The first element of the tuple contains observations, and the second element contains corresponding time stamps for those observations. 
The observations and timestamps are expected to be lists, with one element for each patient. 
Each patient element contains all of the observations or time stamps for that patient. 


On the c++ side, FEAT interprets this into the following format:

```c++
typedef std::map<string, 
                 std::pair<vector<Eigen::ArrayXf>, vector<Eigen::ArrayXf>>
                > LongData;
```

Although a little clunky, the goal is to store patient-specific values in arrays under the hood to allow for as much SIMD optimization as possible when evaluating operators.