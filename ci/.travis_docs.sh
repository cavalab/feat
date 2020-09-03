# build docs if on master branch
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "building doxygen docs"
    cd docs
    doxygen Doxyfile

    echo "doxygen docs build successfully"

    cd ..

    echo "sourcing conda"
    . /home/travis/miniconda/etc/profile.d/conda.sh
    conda activate feat-env

    echo "mkdocs location:"
    which mkdocs
    echo "Building website"
    mkdocs build --verbose --clean
fi
