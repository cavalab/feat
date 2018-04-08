echo "Starting Submission of Classification Jobs"
sleep 1s

echo "Submitting Jobs for Naive Bayes"
sleep 1s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 nb-c


echo "Submitting Jobs for Decision Trees"
sleep 3s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 dt-c


echo "Submitting Jobs for MLP"
sleep 3s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 mlp-c


echo "Submitting Jobs for RF"
sleep 3s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 rf-c

echo "Submitting Jobs for Logreg"
sleep 3s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 logreg-c

echo "Submitting Jobs for Feat"
sleep 3s
./classifier.sh  1 /home/surisr/data-class/ /home/surisr/res-c/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 feat-c

echo "Done Submitting Jobs for Classification Problems... Starting Submission of Regression Jobs"
echo "Submitting Jobs for Feat"

echo "Submitting Jobs for Feat"
sleep 3s
./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 feat-r

echo "Submitting Jobs for Linreg"
sleep 3s
./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 linreg-r

echo "Submitting Jobs for Decision Tree Regressor"
sleep 3s
./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 dt-r

echo "Completed Submission of all jobs...exiting from the program"
