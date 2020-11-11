/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Fernando Iglesias, Yuyu Zhang, Evan Shelhamer, 
 *          Bjoern Esser, Soeren Sonnenburg
 */

#include "MyMulticlassLibLinear.h"

namespace shogun
{

    /** default constructor  */

    CMyMulticlassLibLinear::CMyMulticlassLibLinear() :
	    CLinearMulticlassMachine()
    {
	    init_defaults();
    }

    /** standard constructor
     * @param C C regularization constant value
     * @param features features
     * @param labs labels
     */
    CMyMulticlassLibLinear::CMyMulticlassLibLinear(float64_t C, CDotFeatures* features, CLabels* labs) :
	    CLinearMulticlassMachine(new CMulticlassOneVsRestStrategy(),features,NULL,labs)
    {
	    init_defaults();
	    set_C(C);
    }



    /** destructor */
    CMyMulticlassLibLinear::~CMyMulticlassLibLinear()
    {
	    reset_train_state();
    }



    /** get name */
    const char* CMyMulticlassLibLinear::get_name() const
    {
	    return "MulticlassLibLinear";
    }

    /** set C
     * @param C C value
     */
    inline void CMyMulticlassLibLinear::set_C(float64_t C)
    {
	    ASSERT(C>0)
	    m_C = C;
    }
    
    /** get C
     * @return C value
     */
    inline float64_t CMyMulticlassLibLinear::get_C() const { return m_C; }

    /** set epsilon
     * @param epsilon epsilon value
     */
    inline void CMyMulticlassLibLinear::set_epsilon(float64_t epsilon)
    {
	    ASSERT(epsilon>0)
	    m_epsilon = epsilon;
    }
    
    /** get epsilon
     * @return epsilon value
     */
    inline float64_t CMyMulticlassLibLinear::get_epsilon() const { return m_epsilon; }

    /** set use bias
     * @param use_bias use_bias value
     */
    inline void CMyMulticlassLibLinear::set_use_bias(bool use_bias)
    {
	    m_use_bias = use_bias;
    }
    
    /** get use bias
     * @return use_bias value
     */
    inline bool CMyMulticlassLibLinear::get_use_bias() const
    {
	    return m_use_bias;
    }

    /** set save train state
     * @param save_train_state save train state value
     */
    inline void CMyMulticlassLibLinear::set_save_train_state(bool save_train_state)
    {
	    m_save_train_state = save_train_state;
    }
    
    /** get save train state
     * @return save_train_state value
     */
    inline bool CMyMulticlassLibLinear::get_save_train_state() const
    {
	    return m_save_train_state;
    }

    /** set max iter
     * @param max_iter max iter value
     */
    inline void CMyMulticlassLibLinear::set_max_iter(int32_t max_iter)
    {
	    ASSERT(max_iter>0)
	    m_max_iter = max_iter;
    }
    
    /** get max iter
     * @return max iter value
     */
    inline int32_t CMyMulticlassLibLinear::get_max_iter() const { return m_max_iter; }

    /** reset train state */
    void CMyMulticlassLibLinear::reset_train_state()
    {
	    if (m_train_state)
	    {
		    delete m_train_state;
		    m_train_state = NULL;
	    }
    }

    /** get support vector indices
     * @return support vector indices
     */
    SGVector<int32_t> CMyMulticlassLibLinear::get_support_vectors() const
    {
	    if (!m_train_state)
		    SG_ERROR("Please enable save_train_state option and train machine.\n")

	    ASSERT(m_labels && m_labels->get_label_type() == LT_MULTICLASS)

	    int32_t num_vectors = m_features->get_num_vectors();
	    int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();

	    v_array<int32_t> nz_idxs;
	    nz_idxs.reserve(num_vectors);

	    for (int32_t i=0; i<num_vectors; i++)
	    {
		    for (int32_t y=0; y<num_classes; y++)
		    {
			    if (CMath::abs(m_train_state->alpha[i*num_classes+y])>1e-6)
			    {
				    nz_idxs.push(i);
				    break;
			    }
		    }
	    }
	    int32_t num_nz = nz_idxs.index();
	    nz_idxs.reserve(num_nz);
	    return SGVector<int32_t>(nz_idxs.begin,num_nz);
    }

    /* get the weights for each SVM Subclass
    * @return the vector of weights for each subclass
    */
    vector<SGVector<float64_t>> CMyMulticlassLibLinear::get_w() const
    {
	    vector<SGVector<float64_t>> weights_vector;
	    int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();

	    for (int32_t i=0; i<num_classes; i++)
	    {
		    CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
		    weights_vector.push_back(machine->get_w().clone());
	    }

         return weights_vector;

    }

    void CMyMulticlassLibLinear::set_w(vector<Eigen::VectorXd> wnew)
    {
        
	    int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
        
        for (int32_t i=0; i<num_classes; i++)
	    {
		    CLinearMachine* machine = (CLinearMachine*)m_machines->get_element(i);
		    machine->set_w(SGVector<float64_t>(wnew.at(i)));
	    }
	
    }
    
    bool CMyMulticlassLibLinear::train_machine(CFeatures* data)
    {
	    if (data)
		    set_features((CDotFeatures*)data);

	    ASSERT(m_features)
	    ASSERT(m_labels && m_labels->get_label_type()==LT_MULTICLASS)
	    ASSERT(m_multiclass_strategy)
	    init_strategy();

	    int32_t num_vectors = m_features->get_num_vectors();
	    int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
	    int32_t bias_n = m_use_bias ? 1 : 0;

	    liblinear_problem mc_problem;
	    mc_problem.l = num_vectors;
	    mc_problem.n = m_features->get_dim_feature_space() + bias_n;
	    mc_problem.y = SG_MALLOC(float64_t, mc_problem.l);
	    for (int32_t i=0; i<num_vectors; i++)
		    mc_problem.y[i] = ((CMulticlassLabels*) m_labels)->get_int_label(i);

	    mc_problem.x = m_features;
	    mc_problem.use_bias = m_use_bias;

	    SGMatrix<float64_t> w0 = obtain_regularizer_matrix();

	    if (!m_train_state)
		    m_train_state = new mcsvm_state();

	    float64_t* C = SG_MALLOC(float64_t, num_vectors);
	    for (int32_t i=0; i<num_vectors; i++)
		    C[i] = m_C;

	    Solver_MCSVM_CS solver(&mc_problem,num_classes,C,w0.matrix,m_epsilon,
			           m_max_iter,m_max_train_time,m_train_state);
	    solver.solve();

	    m_machines->reset_array();
	    for (int32_t i=0; i<num_classes; i++)
	    {
		    CLinearMachine* machine = new CLinearMachine();
		    SGVector<float64_t> cw(mc_problem.n-bias_n);

		    for (int32_t j=0; j<mc_problem.n-bias_n; j++)
			    cw[j] = m_train_state->w[j*num_classes+i];

		    machine->set_w(cw);

		    if (m_use_bias)
			    machine->set_bias(m_train_state->w[(mc_problem.n-bias_n)*num_classes+i]);

		    m_machines->push_back(machine);
	    }

	    if (!m_save_train_state)
		    reset_train_state();

	    SG_FREE(C);
	    SG_FREE(mc_problem.y);

	    return true;
    }


    /** obtain regularizer (w0) matrix */
    SGMatrix<float64_t> CMyMulticlassLibLinear::obtain_regularizer_matrix() const
    {
            return SGMatrix<float64_t>();
    }

    void CMyMulticlassLibLinear::init_defaults()
    {
	    set_C(1.0);
	    set_epsilon(1e-2);
	    set_max_iter(10000);
	    set_use_bias(false);
	    set_save_train_state(false);
	    m_train_state = NULL;
    }

    void CMyMulticlassLibLinear::register_parameters()
    {
	    SG_ADD(&m_C, "m_C", "regularization constant",MS_AVAILABLE);
	    SG_ADD(&m_epsilon, "m_epsilon", "tolerance epsilon",MS_NOT_AVAILABLE);
	    SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations",MS_NOT_AVAILABLE);
	    SG_ADD(&m_use_bias, "m_use_bias", "indicates whether bias should be used",MS_NOT_AVAILABLE);
	    SG_ADD(&m_save_train_state, "m_save_train_state", "indicates whether bias should be used",MS_NOT_AVAILABLE);
    }
}

