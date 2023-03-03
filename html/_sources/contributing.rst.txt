Contributing
============

Please follow the `Github flow`_ guidelines for contributing to this
project.

In general, this is the approach:

-  Fork the repo into your own repository and clone it locally.

::

   git clone https://github.com/my_user_name/feat

-  Have an idea for a code change. Checkout a new branch with an
   appropriate name.

::

   git checkout -b my_new_change

-  Make your changes.
-  Commit your changes to the branch.

::

   git commit -m "adds my new change"

-  Check that your branch has no conflict with Feat’s master branch by
   merging the master branch from the upstream repo.

::

   git remote add upstream https://github.com/lacava/feat
   git fetch upstream
   git merge upstream/master

-  Fix any conflicts and commit.

::

   git commit -m "Merges upstream master"

-  Push the branch to your forked repo.

::

   git push origin my_new_change

-  Go to either Github repo and make a new Pull Request for your forked
   branch. Be sure to reference any relevant issues.

.. _Github flow: https://guides.github.com/introduction/flow/
