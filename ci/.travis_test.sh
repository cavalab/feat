cd ./build
./tests

echo "Running Feat Wrapper Tests"

cd ../python
python wrappertest.py -v 1
