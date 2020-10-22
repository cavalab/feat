/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Fernando Iglesias, Yuyu Zhang, Evan Shelhamer, 
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef _MULTICLASSLIBLINEAR_H___
#define _MULTICLASSLIBLINEAR_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/optimization/liblinear/shogun_liblinear.h>
#include <shogun/lib/config.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/v_array.h>
#include <shogun/lib/Signal.h>
#include <shogun/labels/MulticlassLabels.h>
#include <vector>
#include <Eigen/Dense>
using std::vector;


namespace shogun
{

    /** @brief multiclass LibLinear wrapper. Uses Crammer-Singer
        formulation and gradient descent optimization algorithm
        implemented in the LibLinear library. Regularized bias
        support is added using stacking bias 'feature' to
        hyperplanes normal vectors.

        In case of small changes of C or particularly epsilon
        this class provides ability to save whole liblinear
        training state (i.e. W vector, gradients, etc) and re-use
        on next train() calls. This capability could be
        enabled using set_save_train_state() method. Train
        state can be forced to clear using
        reset_train_state() method.
     */
    class CMyMulticlassLibLinear : public CLinearMulticlassMachine
    {
	    public:
		    
		    MACHINE_PROBLEM_TYPE(PT_MULTICLASS);
		    
		    CMyMulticlassLibLinear();

		    /** standard constructor
		     * @param C C regularization constant value
		     * @param features features
		     * @param labs labels
		     */
		    CMyMulticlassLibLinear(float64_t C, CDotFeatures* features, CLabels* labs);

		    /** destructor */
		    ~CMyMulticlassLibLinear();

		    /** get name */
		    virtual const char* get_name() const;

		    /** set C
		     * @param C C value
		     */
		    inline void set_C(float64_t C);
		    
		    /** get C
		     * @return C value
		     */
		    inline float64_t get_C() const;

		    /** set epsilon
		     * @param epsilon epsilon value
		     */
		    inline void set_epsilon(float64_t epsilon);
		    
		    /** get epsilon
		     * @return epsilon value
		     */
		    inline float64_t get_epsilon() const;

		    /** set use bias
		     * @param use_bias use_bias value
		     */
		    inline void set_use_bias(bool use_bias);
		    
		    /** get use bias
		     * @return use_bias value
		     */
		    inline bool get_use_bias() const;

		    /** set save train state
		     * @param save_train_state save train state value
		     */
		    inline void set_save_train_state(bool save_train_state);
		    
		    /** get save train state
		     * @return save_train_state value
		     */
		    inline bool get_save_train_state() const;

		    /** set max iter
		     * @param max_iter max iter value
		     */
		    inline void set_max_iter(int32_t max_iter);
		    
		    /** get max iter
		     * @return max iter value
		     */
		    inline int32_t get_max_iter() const;

		    /** reset train state */
		    void reset_train_state();

		    /** get support vector indices
		     * @return support vector indices
		     */
		    SGVector<int32_t> get_support_vectors() const;
	
		    /* get the weights for each SVM Subclass
		    * @return the vector of weights for each subclass
		    */
		    vector<SGVector<float64_t>> get_w() const;
            void set_w(vector<Eigen::VectorXd> wnew);

    protected:

		    /** train machine */
		    bool train_machine(CFeatures* data);

		    /** obtain regularizer (w0) matrix */
		    SGMatrix<float64_t> obtain_regularizer_matrix() const;

    private:

		    void init_defaults();

		    void register_parameters();


    protected:

		    /** regularization constant for each machine */
		    float64_t m_C;

		    /** tolerance */
		    float64_t m_epsilon;

		    /** max number of iterations */
		    int32_t m_max_iter;

		    /** use bias */
		    bool m_use_bias;

		    /** save train state */
		    bool m_save_train_state;

		    /** solver state */
		    mcsvm_state* m_train_state;
    };
}
#endif

