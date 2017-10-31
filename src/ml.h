/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef ML_H
#define ML_H

//external includes
#include <shogun/base/init.h>
#include <shogun/base/Parallel.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/machine/RandomForest.h>

// stuff being used
using std::string;
using std::dynamic_pointer_cast;
using std::cout;
namespace sh = shogun;

namespace FT{
	
	/*!
     * @class ML
     * @brief class that specifies the machine learning algorithm to pair with Fewtwo. 
     */
    class ML 
    {
        public:
        	
            ML(string ml, bool classification)
            {
                /*!
                 * use string to specify a desired ML algorithm from shogun.
                 */
                
                sh::init_shogun_with_defaults();  // initialize shogun if needed
                type = ml;
                auto prob_type = sh::EProblemType::PT_REGRESSION;
                
                if (classification)
                    prob_type = sh::EProblemType::PT_MULTICLASS;                        
                
                
                if (!ml.compare("LeastAngleRegression"))
                    p_est = make_shared<sh::CLeastAngleRegression>();
                
                else if (!ml.compare("RandomForest")){
                    p_est = make_shared<sh::CRandomForest>();
                    dynamic_pointer_cast<sh::CRandomForest>(p_est)->
                                                               set_machine_problem_type(prob_type);
                }
                else if (!ml.compare("CART")){
                    p_est = make_shared<sh::CCARTree>();
                    dynamic_pointer_cast<sh::CCARTree>(p_est)->set_machine_problem_type(prob_type);
                }

                else if (!ml.compare("LinearRidgeRegression"))
                    p_est = make_shared<sh::CLinearRidgeRegression>();                   
                
                else
                    std::cerr << "'" + ml + "' is not a valid ml choice\n";
                
            }
        
            ~ML(){ sh::exit_shogun(); }

            // return vector of weights for model. 
            vector<double> get_weights();

            shared_ptr<sh::CMachine> p_est;
            string type;
    };


    vector<double> ML::get_weights()
    {    
        /*!
         * return weight vector from model.
         */
        vector<double> w;
        
        if (!type.compare("LeastAngleRegression") || !type.compare("LinearRidgeRegression"))
        {
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
                
        } 
        return w;
    }


}


#endif
