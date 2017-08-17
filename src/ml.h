/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ML_H
#define ML_H

//external includes
#include <shogun/base/init.h>
#include <shogun/machine/Machine.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/multiclass/tree/C45ClassifierTree.h>
#include <shogun/multiclass/tree/C45TreeNodeData.h>
#include <shogun/machine/RandomForest.h>

// stuff being used
using std::string;
namespace sh = shogun;

namespace FT{

    class ML 
    {
        /* class that specifies the machine learning algorithm to pair with Fewtwo. 
         */

        public:
        
            ML(string ml, bool init=true)
            {
                /* use string to specify a desired ML algorithm from shogun. */
                                
                if (init) sh::init_shogun_with_defaults();  // initialize shogun if needed

                if (!ml.compare("LeastAngleRegression"))
                    p_est = make_shared<sh::CLeastAngleRegression>();
                
                else if (!ml.compare("RandomForest"))
                    p_est = make_shared<sh::CRandomForest>();
                
                else if (!ml.compare("C45"))
                    p_est = make_shared<sh::CTreeMachine<sh::C45TreeNodeData>>();

                else if (!ml.compare("LinearRidgeRegression"))
                    p_est = make_shared<sh::CLinearRidgeRegression>();                   
                
                else
                    std::cerr << "'" + ml + "' is not a valid ml choice\n";
                
            }
        
            ~ML(){ sh::exit_shogun(); }

      
            shared_ptr<sh::CMachine> p_est;
    };
}

#endif
