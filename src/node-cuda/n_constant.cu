/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "cuda_utils.h"
#include "n_constant.h"

namespace FT{
   		
    void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
        		if (otype == 'b')
                    stack_b.push_back(ArrayXb::Constant(X.cols(),int(b_value)));
                else 	
                    stack_f.push_back(ArrayXd::Constant(X.cols(),d_value));
            }

}	
#endif

