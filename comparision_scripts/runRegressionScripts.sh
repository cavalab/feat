#echo "Submitting Jobs for Feat"
#sleep 3s
#./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 feat-r

echo "Submitting Jobs for Linreg"
sleep 3s
./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 linreg-r

echo "Submitting Jobs for Decision Tree Regressor"
sleep 3s
./classifier.sh  1 /home/surisr/data/ /home/surisr/res-r/ /home/surisr/logs/ /home/surisr/errs/ 10 0.75 dt-r

echo "Completed Submission of all jobs...exiting from the program"
