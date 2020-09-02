# build docs if on master branch
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "building doxygen docs"
    cd docs
    doxygen Doxyfile

    echo "docs build successfully"

    cd ..

    echo "sourcing conda"
    . /home/travis/miniconda/etc/profile.d/conda.sh
    conda activate feat-env

    echo "Building mkdocs"
    mkdocs build --verbose --clean
fi
