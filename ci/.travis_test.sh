gcc --version
g++ --version
cd ./tests
cmake -DCMAKE_C_COMPILER=$OUR_CC -DCMAKE_CXX_COMPILER=$OUR_CXX -DCMAKE_BUILD_TYPE=$BUILD_TYPE .
make
./runTests
