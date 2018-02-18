# install gcc 4.8
#echo "installing gcc 4.8..."
#sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
#sudo apt-get update -qq
##if [ "$CXX" = "g++" ]; then sudo apt-get install -qq g++-4.8; fi
##if [ "$CXX" = "g++" ]; then export CXX="g++-4.8" CC="gcc-4.8"; fi
#sudo apt-get install -qq g++-4.8;
#export CXX="g++-4.8"


#echo "we need cmake 3 or greater. installing..."
#curl -sSL https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz | tar -xz
#export PATH=/cmake-3.5.2-Linux-x86_64/bin/:$PATH
#cmake --version

#installing eigen 3
 #sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz" -O- | sudo tar xvz -C /usr/include/
echo "installing eigen..."
wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
tar xzf 3.3.4.tar.gz 
mkdir eigen-3.3.4 
mv eigen-eigen*/* eigen-3.3.4

export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"
echo "EIGEN3_INCLUDE_DIR set to $EIGEN3_INCLUDE_DIR"
#_______________________________________________
  
#installing shogun library
#echo "installing shogun..." 
#sudo apt-get install -qq software-properties-common lsb-release
#sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
#sudo add-apt-repository ppa:shogun-toolbox/nightly -y
#sudo apt-get update -y
#sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
#sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev
## sudo apt-get install -qq --force-yes --no-install-recommends python-shogun
#sudo dpkg-query -l '*shogun*'
echo "installing shogun via conda..."
wget http://repo.continuum.io/miniconda/Miniconda-3.9.1-Linux-x86_64.sh \
        -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
#export PATH=/root/miniconda/bin:$PATH
conda update --yes conda
conda install --yes -c conda-forge shogun-cpp
export SHOGUN_LIB=/home/travis/miniconda/lib/
export SHOGUN_DIR=/home/travis/miniconda/include/

#building and installing google tests
# sudo apt-get install cmake
echo "installing google test"
#sudo apt-get install libgtest-dev
#sudo dpkg -L libgtest-dev 
##gtest_path=$(dpkg -L ligtest-dev | awk '{print $2}')
##echo "gtest_path set to $gtest_path"
#old_path=$(pwd)
#echo "building google test.."
#cd /usr/src/gtest; echo "changed to $(pwd)"
#echo "ls"
#ls
##cat CMakeLists.txt
#sudo mkdir build
#sudo cd build
#sudo cmake .. 
#sudo make
#echo "cp library to /usr/lib"
#sudo cp *.a /usr/lib
#echo "back to $old_path.."
#cd $old_path; pwd
git clone https://github.com/google/googletest
GTEST_DIR="$(pwd)/googletest"
cd googletest
mkdir build
cd build
echo "running cmake.."
cmake ..
make
echo "copying library..."
ls
cp *.a ../../tests/

echo "installing feat..."
cd ../..; pwd
mkdir build;
cd build; pwd 

cmake -DEIGEN_DIR=ON -DSHOGUN_DIR=ON ..

cd ..
make -C build VERBOSE=1
echo "running feat.."
./build/feat examples/d_enc.csv
