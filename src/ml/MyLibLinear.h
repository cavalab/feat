/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef MYLIBLINEAR_H
#define MYLIBLINEAR_H

#include <shogun/classifier/svm/LibLinear.h>

namespace shogun {
    class CMyLibLinear : public CLibLinear
    {
        public:

            /** default constructor  */
            /* CMyLibLinear(); */

            /** constructor
             *
             * @param liblinear_solver_type liblinear_solver_type
             */
            CMyLibLinear(LIBLINEAR_SOLVER_TYPE liblinear_solver_type);

            /** constructor (using L2R_L1LOSS_SVC_DUAL as default)
             *
             * @param C constant C
             * @param traindat training features
             * @param trainlab training labels
             */

            /** destructor */
            /* virtual ~CMyLibLinear(); */

        
            /** WGL: predict probability function, adapted from liblinear package */
            void predict_probability(CLabels* labels, const SGVector<float64_t>& outputs);
            void set_probabilities(CLabels* labels, CFeatures* data);
		private:
			void init();
    };
} /* namespace shogun  */
#endif //_LIBLINEAR_H___


