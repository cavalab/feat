# feat c++ tests
echo "==========\nc++ tests\n=========="
./build/tests

echo "==========\npython tests\n=========="
# python tests
echo "sourcing conda"
. "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
echo "reactivating test-environment"
/home/travis/miniconda/condabin/conda activate test-environment
/home/travis/miniconda/condabin/conda info -a
echo "running wrapper test"
/home/travis/miniconda/bin/python3 python/wrappertest.py -v 1


