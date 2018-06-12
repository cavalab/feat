/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "MyLibLinear.h"

namespace shogun {
    
    
    CMyLibLinear::CMyLibLinear(LIBLINEAR_SOLVER_TYPE l) : CLibLinear() 
    {
	    set_liblinear_solver_type(l);
    }
    
    void CMyLibLinear::init()
    {
	    set_liblinear_solver_type(L2R_L1LOSS_SVC_DUAL);
	    set_bias_enabled(false);
	    set_C(1, 1);
	    set_max_iterations();
	    set_epsilon(1e-5);
	    /** Prevent default bias computation*/
	    set_compute_bias(false);

	    SG_ADD(&C1, "C1", "C Cost constant 1.", MS_AVAILABLE);
	    SG_ADD(&C2, "C2", "C Cost constant 2.", MS_AVAILABLE);
	    SG_ADD(
	        &use_bias, "use_bias", "Indicates if bias is used.", MS_NOT_AVAILABLE);
	    SG_ADD(&epsilon, "epsilon", "Convergence precision.", MS_NOT_AVAILABLE);
	    SG_ADD(
	        &max_iterations, "max_iterations", "Max number of iterations.",
	        MS_NOT_AVAILABLE);
	    SG_ADD(&m_linear_term, "linear_term", "Linear Term", MS_NOT_AVAILABLE);
	    SG_ADD(
	        (machine_int_t*)&liblinear_solver_type, "liblinear_solver_type",
	        "Type of LibLinear solver.", MS_NOT_AVAILABLE);
    }
    
    // WGL: updated definitions to allow probabilities to be calculated
    void CMyLibLinear::set_probabilities(CLabels* labels, CFeatures* data)
    {
        // get probabilities for each output
        if (liblinear_solver_type == L1R_LR || 
             liblinear_solver_type == L2R_LR_DUAL || 
             liblinear_solver_type == L2R_LR )
        {
	        SGVector<float64_t> outputs = apply_get_outputs(data);
            predict_probability(labels,outputs);
        }
        else
            dynamic_cast<CBinaryLabels*>(labels)->scores_to_probabilities();
    }

    void CMyLibLinear::predict_probability(CLabels* labels, const SGVector<float64_t>& outputs)
    {
            /* cout << "outputs:\n"; */
            /* outputs.display_vector(); */

            for (int i = 0; i < outputs.size(); ++i)
            {
                labels->set_value(1/(1+exp(-outputs[i])),i);
            }
    }

} /* namespace shogun  */


