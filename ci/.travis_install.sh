##############################
# conda setup and installation
##############################

echo "installing shogun and eigen via conda..."
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
. "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
echo "creating conda environment"
conda config --set always_yes yes --set changeps1 no
conda env create -f ci/test-environment.yml
# conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION -c conda-forge shogun-cpp eigen json-c=0.12.1-0 cython scikit-learn pandas
echo "activating test-environment"
conda activate feat-env

# install packages for the docs
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "installing mkdocs"
    conda env update feat-env -f ci/docs-environment.yml
    echo "mkdocs version"
    mkdocs --version
fi

which conda
conda info -a

echo "printing conda environment"
conda-env export

echo "python path is..."
which python
python --version

echo "cython path is..."
which cython

# set environment variables for eigen and shogun includes
export EIGEN3_INCLUDE_DIR="$CONDA_PREFIX/include/eigen3/"
echo "EIGEN3_INCLUDE_DIR set to $EIGEN3_INCLUDE_DIR"
export SHOGUN_LIB="$CONDA_PREFIX/lib/"
echo "SHOGUN_LIB set to $SHOGUN_LIB"
export SHOGUN_DIR="$CONDA_PREFIX/include/"
echo "SHOGUN_DIR set to $SHOGUN_DIR"


####################################
#building and installing google tests
####################################
echo "installing google test"
old_path=$(pwd)

echo "building google test.."
cd /usr/src/gtest; echo "changed to $(pwd)"
echo "which cmake?"
which cmake
sudo cmake CMakeLists.txt

sudo make
echo "cp library to /usr/lib"
sudo cp *.a /usr/lib
echo "back to $old_path.."
cd $old_path; pwd

###################
# feat installation
###################
echo "configuring feat..."
./configure tests
echo "install feat..."
./install -j 4 y
