# build docs if on master branch
if [ "$TRAVIS_BRANCH" = "master" ]
then
    echo "building doxygen docs"
    cd docs
    mkdir doxygen_site/
    doxygen Doxyfile

    echo "doxygen docs build successfully"

    echo "sourcing conda"
    . /home/travis/miniconda/etc/profile.d/conda.sh
    conda activate feat-env

    echo "building sphinx docs"
    make html
fi
