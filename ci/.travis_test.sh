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
conda activate feat-env
conda info -a
echo "running wrapper test"
python tests/wrappertest.py -v 0


# test example notebooks if on master branch
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "testing notebooks"
    python tests/nb_tests.py
fi
