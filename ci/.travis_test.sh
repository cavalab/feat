echo "=========\njust a quick run====="
./build/feat docs/examples/data/d_enc.csv -rs 42 -g 2 -p 5

# feat c++ tests
echo "==========\nc++ tests\n=========="
./build/tests

echo "==========\npython tests\n=========="
# python tests
echo "sourcing conda"
. /home/travis/miniconda/etc/profile.d/conda.sh
which conda

echo "reactivating test"
conda activate test
conda info -a
echo "running wrapper test"
python tests/wrappertest.py -v 1


