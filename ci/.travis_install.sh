echo "installing cmake"
# sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
# sudo apt-get update -y
sudo apt-get install cmake
echo "cmake version:"
cmake --version
echo "sudo cmake version:"
sudo cmake --version

echo "installing pip"
sudo apt install python3-pip

echo "installing mkdocs"
sudo pip3 install mkdocs==1.0.4 mkdocs-material pymdown-extensions pygments

echo "mkdocs version"
mkdocs --version

echo "installing eigen..."
#wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2"
tar xvjf 3.3.4.tar.bz2 
mkdir eigen-3.3.4 
mv eigen-eigen*/* eigen-3.3.4

export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"
echo "EIGEN3_INCLUDE_DIR set to $EIGEN3_INCLUDE_DIR"
#_______________________________________________
  
echo "installing shogun via conda..."
wget http://repo.continuum.io/miniconda/Miniconda-3.9.1-Linux-x86_64.sh \
        -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH

conda update --yes conda
conda install --yes -c conda-forge shogun-cpp

# the new version of json-c seems to be missing a fn shogun is linked to; 
# force install of older version
conda install --yes json-c=0.12.1-0

export SHOGUN_LIB=/home/travis/miniconda/lib/
export SHOGUN_DIR=/home/travis/miniconda/include/
# commending out the following installs which should be triggered
# by call to setup.py
echo "installing cython using conda..."
conda install --yes cython

echo "installing scikit-learn via conda..."
conda install --yes scikit-learn

echo "installing pandas via conda..."
conda install --yes pandas

#building and installing google tests
echo "installing google test"
sudo apt-get install libgtest-dev
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

echo "installing feat..."
mkdir build;
cd build; pwd 

cmake -DTEST=ON -DEIGEN_DIR=ON -DSHOGUN_DIR=ON ..

cd ..
make -C build VERBOSE=1
echo "running feat.."

./build/feat docs/examples/data/d_enc.csv -rs 42 -g 2 -p 5

echo "python path is..."
which python

echo "cython path is..."
which cython

echo "upgrading pip..."
pip install --upgrade pip

echo "installing wrapper"
cd ./python
python setup.py install

echo "copying wrapper test to the python folder"
sudo cp ../tests/wrappertest.py ./ #Copy the file to python folder

