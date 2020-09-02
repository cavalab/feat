echo "python path is..."
which python
python --version

echo "cython path is..."
which cython
echo "installing cmake"
# sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
# sudo apt-get update -y
# sudo apt-get install cmake
echo "cmake version:"
cmake --version
echo "sudo cmake version:"
sudo cmake --version

# echo "installing pip"
# sudo apt install python3-pip
# echo "installing setuptools"
# sudo -H pip3 install setuptools
# echo "installing wheel"
# sudo -H pip3 install wheel

#wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
# wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2"
# tar xvjf 3.3.4.tar.bz2
# mkdir eigen-3.3.4
# mv eigen-eigen*/* eigen-3.3.4

##########CONDA##############

echo "installing shogun and eigen via conda..."
wget http://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
. "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
echo "creating conda environment"
conda config --set always_yes yes --set changeps1 no
# cython=0.29.12
conda create -c conda-forge -q -n test-environment python=3.7 shogun-cpp=6.1.3 eigen=3.3.5 json-c=0.12.1-0 cython scikit-learn pandas wheel setuptools=41.0.1
# conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION -c conda-forge shogun-cpp eigen json-c=0.12.1-0 cython scikit-learn pandas
echo "activating test-environment"
conda activate test-environment

# install packages for the docs
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "installing mkdocs"
    conda install mkdocs==1.1 mkdocs-material pymdown-extensions pygments
    echo "mkdocs version"
    mkdocs --version
fi

which conda
conda info -a

# conda update --yes conda
# conda install --yes -c conda-forge shogun-cpp eigen

# export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"
# conda install -c conda-forge eigen

# the new version of json-c seems to be missing a fn shogun is linked to;
# force install of older version
# conda install --yes json-c=0.12.1-0

# commending out the following installs which should be triggered
# by call to setup.py
# echo "installing cython using conda..."
# conda install --yes cython

# echo "installing scikit-learn via conda..."
# conda install --yes scikit-learn

# echo "installing pandas via conda..."
# conda install --yes pandas

echo "printing conda environment"
conda-env export

echo "python path is..."
which python
python --version

echo "cython path is..."
which cython
##########CONDA##############

# set environment variables for eigen and shogun includes
export EIGEN3_INCLUDE_DIR="$HOME/miniconda/envs/test-environment/include/eigen3/"
echo "EIGEN3_INCLUDE_DIR set to $EIGEN3_INCLUDE_DIR"

export SHOGUN_LIB=/home/travis/miniconda/envs/test-environment/lib/
export SHOGUN_DIR=/home/travis/miniconda/envs/test-environment/include/


#building and installing google tests
echo "installing google test"
# sudo apt-get install libgtest-dev
old_path=$(pwd)

echo "building google test.."
cd /usr/src/gtest; echo "changed to $(pwd)"
echo "ls"
ls
sudo cmake CMakeLists.txt

sudo make
echo "cp library to /usr/lib"
sudo cp *.a /usr/lib
echo "back to $old_path.."
cd $old_path; pwd

###################
# feat installation
###################
echo "installing feat..."
mkdir build;
cd build; pwd
cmake -DTEST=ON -DEIGEN_DIR=ON -DSHOGUN_DIR=ON ..
cd ..
make -C build VERBOSE=1

echo "installing wrapper"
cd ./python
python setup.py install
cd ..

echo "copying wrapper test to the python folder"
sudo cp tests/wrappertest.py python/ #Copy the file to python folder
