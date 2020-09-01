# feat c++ tests
echo "==========\nc++ tests\n=========="
./build/tests

echo "==========\npython tests\n=========="
# python tests
echo "reactivating test-environment"
conda activate test-environment
conda info -a
/home/travis/miniconda/bin/python3 python/wrappertest.py -v 1


