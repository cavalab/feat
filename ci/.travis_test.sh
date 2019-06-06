# feat c++ tests
echo "==========\nc++ tests\n=========="
./build/tests

echo "==========\npython tests\n=========="
# python tests
/home/travis/miniconda/bin/python python/wrappertest.py -v 1


