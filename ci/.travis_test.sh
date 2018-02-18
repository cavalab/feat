cd ./tests
cmake -DEIGEN_DIR=ON -DSHOGUN_DIR=ON .
make
./runTests
