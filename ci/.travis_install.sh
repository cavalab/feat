
echo "installing eigen..."
wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
tar xzf 3.3.4.tar.gz 
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
export SHOGUN_LIB=/home/travis/miniconda/lib/
export SHOGUN_DIR=/home/travis/miniconda/include/

echo "installing scikit-learn via conda..."
conda install --yes scikit-learn

echo "installing pandas via conda..."
conda install --yes pandas

#building and installing google tests
sudo apt-get install cmake
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
./build/feat examples/d_enc.csv

cd ./python
sudo python setup.py install build_ext --inplace

#_____Run the Python Tests for the wrapper_____#
cd ../tests
sudo cp wrappertest.py ../python/ #Copy the file to python folder
cd ../python/
sudo python wrappertest.py -v 1
