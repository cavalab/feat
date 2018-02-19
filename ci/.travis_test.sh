export EIGEN3_INCLUDE_DIR="$(pwd)/eigen-3.3.4/"
echo "EIGEN3_INCLUDE_DIR set to $EIGEN3_INCLUDE_DIR"

export SHOGUN_LIB=/home/travis/miniconda/lib/
export SHOGUN_DIR=/home/travis/miniconda/include/

echo "SHOGUN_LIB set to $SHOGUN_LIB"
echo "SHOGUN_DIR set to $SHOGUN_DIR"

cd ./tests
cmake -DEIGEN_DIR=ON -DSHOGUN_DIR=ON .
make
./runTests
