# FEAT

<!-- start overview -->

[![Build Status](https://github.com/cavalab/feat/actions/workflows/ci.yml/badge.svg)](https://github.com/cavalab/feat/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/lacava/feat/blob/master/LICENSE)

**FEAT** is a feature engineering automation tool that learns new representations of raw data 
to improve classifier and regressor performance. 
The underlying methods use Pareto optimization and evolutionary computation to search the space of possible transformations.

FEAT wraps around a user-chosen ML method and provides a set of representations that give the best 
performance for that method. 
Each individual in FEAT's population is its own data representation. 

FEAT uses the [Shogun C++ ML toolbox](http://shogun.ml) to fit models. 

Check out the [documentation](https://cavalab.org/feat) for installation and examples. 

## References

1. La Cava, W., Singh, T. R., Taggart, J., Suri, S., & Moore, J. H.. Learning concise representations for regression by evolving networks of trees. ICLR 2019. [arxiv:1807.0091](https://arxiv.org/abs/1807.00981)

2. La Cava, W. & Moore, Jason H. (2020).
Genetic programming approaches to learning fair classifiers.
GECCO 2020.
**Best Paper Award**.
[ACM](https://dl.acm.org/doi/abs/10.1145/3377930.3390157),
[arXiv](https://arxiv.org/abs/2004.13282),
[experiments](https://github.com/lacava/fair_gp)

3. La Cava, W., Lee, P.C., Ajmal, I., Ding, X., Cohen, J.B., Solanki, P., Moore, J.H., and Herman, D.S (2021).
Application of concise machine learning to construct accurate and interpretable EHR computable phenotypes.
In Review.
[medRxiv](https://www.medrxiv.org/content/10.1101/2020.12.12.20248005v2),
[experiments](https://bitbucket.org/hermanlab/ehr_feat/)



## Contact

Maintained by William La Cava (william.lacava at childrens.harvard.edu)

  
## Acknowledgments

Special thanks to these contributors:
  - Tilak Raj Singh @tilakhere 
  - Srinivas Suri @srinu634
  - James P Taggert @JPT2
  - Daniel Herman 
  - Paul Lee

This work is supported by grant K99-LM012926 and R00-LM012926 from the National Library of Medicine. 
FEAT is being developed to learn clinical diagnostics in the [Cava Lab at Harvard Medical School](http://cavalab.org). 

## License

GNU GPLv3, see [LICENSE](https://github.com/cavalab/feat/blob/master/LICENSE)

<!-- end overview -->

# Installation

<!-- start installation -->

To see our installation process from scratch, check out the [Github
actions
workflow](http://github.com/lacava/feat/blob/master/.github/workflows/ci.yml).

## Dependencies

Feat uses [cmake](https://cmake.org/) to build. It also depends on the
[Eigen](http://eigen.tuxfamily.org) matrix library for C++ as well as
the [Shogun](http://shogun.ml) ML library. Both come in packages on
conda that should work across platforms.

## Install in a Conda Environment

The easiest option for install is to use the [conda environment we
provide](http://github.com/lacava/feat/blob/master/environment.yml).
Then the build process is the following:

``` bash
git clone https://github.com/lacava/feat # clone the repo
cd feat # enter the directory
conda env create
conda activate feat
pip install .
```

If you want to roll your own with the dependencies, some other options
are shown below. In this case, you need to tell the
[configure]{.title-ref} script where Shogun and Eigen are. Edit this
lines:

``` bash
export SHOGUN_LIB=/your/shogun/lib/
export SHOGUN_DIR=/your/shugn/include/
export EIGEN3_INCLUDE_DIR=/your/eigen/eigen3/
```

If you need Eigen and Shogun and don\'t want to use conda, follow these
instructions.

## Eigen

Eigen is a header only package. We need Eigen 3 or greater.

### Debian/Ubuntu

On Debian systems, you can grab the package:

> sudo apt-get install libeigen3-dev

You can also download the headers and put them somewhere. Then you just
have to tell cmake where they are with the environmental variable
`EIGEN3_INCLUDE_DIR`. Example:

``` bash
# Eigen 3.3.4
wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
tar xzf 3.3.4.tar.gz 
mkdir eigen-3.3.4 
mv eigen-eigen*/* eigen-3.3.4
# set an environmental variable to tell cmake where Eigen is
export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"
```

## Shogun

You don\'t have to compile Shogun, just download the binaries. [Their
install guide is
good.](https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INSTALL.md#binaries)
We\'ve listed two of the options here.

### Debian/Ubuntu

You can also get the Shogun packages:

``` bash
sudo add-apt-repository ppa:shogun-toolbox/nightly -y
sudo apt-get update -y
sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev
```

<!-- end installation -->

## Running the tests

<!-- start tests -->

*(optional)* If you want to run the c++ tests, you need to install [Google
Test](https://github.com/google/googletest). A useful guide to doing so
is available
[here](https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/).
Then you can use cmake to build the tests. From the repo root,

``` bash
./configure tests   # builds the test Makefile
make -C build tests # compiles the tests
./build/tests # runs the tests
```

For the python tests, run 

```python
python tests/wrappertest.py
```

<!-- end tests -->

## Contributing

<!-- start contributing -->

Please follow the [Github
flow](https://guides.github.com/introduction/flow/) guidelines for
contributing to this project.

In general, this is the approach:

-   Fork the repo into your own repository and clone it locally.

    ```
    git clone https://github.com/my_user_name/feat
    ```

-   Have an idea for a code change. Checkout a new branch with an
    appropriate name.

    ```
    git checkout -b my_new_change
    ```

-   Make your changes.
-   Commit your changes to the branch.

    ```
    git commit -m "adds my new change"
    ```

-   Check that your branch has no conflict with Feat's master branch by
    merging the master branch from the upstream repo.

    ```
    git remote add upstream https://github.com/lacava/feat
    git fetch upstream
    git merge upstream/master
    ```

-   Fix any conflicts and commit.

    ```
    git commit -m "Merges upstream master"
    ```

-   Push the branch to your forked repo.

    ```
    git push origin my_new_change
    ```

-   Go to either Github repo and make a new Pull Request for your forked
    branch. Be sure to reference any relevant issues.

<!-- end contributing -->