
#installing eigen 3
 #sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz" -O- | sudo tar xvz -C /usr/include/
sudo wget "http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz"
sudo tar xvzf 3.3.4.tar.gz
export EIGEN3_INCLUDE_DIR="3.3.4/eigen3/"

#_______________________________________________
  
#installing shogun library
 sudo apt-get install -qq software-properties-common lsb-release
 sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
 sudo add-apt-repository ppa:shogun-toolbox/stable -y
 sudo apt-get update -y
 sudo apt-get install -qq --force-yes --no-install-recommends libshogun18
# sudo apt-get install -qq --force-yes --no-install-recommends libshogun-dev
# sudo apt-get install -qq --force-yes --no-install-recommends python-shogun
 sudo dpkg-query -l '*shogun*'
  
v#building and installing google tests
# sudo apt-get install cmake
# sudo apt-get install libgtest-dev
 
 cmake --version
 pwd
 cd /usr/src/gtest; pwd
 sudo cmake CMakeLists.txt
 sudo make
 sudo cp *.a /usr/lib
 cd /home/travis/build/lacava/feat; pwd

 cd build; pwd
 
 cmake -DEIGEN_DIR=ON ..

 make -C build VERBOSE=1

 ./build/feat examples/d_enc.csv
#_________________________________________________________

