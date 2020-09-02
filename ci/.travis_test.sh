# feat c++ tests
echo "==========\nc++ tests\n=========="
./build/tests

echo "==========\npython tests\n=========="
# python tests
echo "sourcing conda"
. /home/travis/miniconda/etc/profile.d/conda.sh
which conda

echo "reactivating test-environment"
conda activate test-environment
conda info -a
echo "running wrapper test"
/home/travis/miniconda/bin/python3 python/wrappertest.py -v 1


