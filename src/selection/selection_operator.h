/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

namespace FT{

    /*!
     * @class SelectionOperator
     */ 
    struct SelectionOperator 
    {
        /*!
         * base class for selection operators.
         */

        bool survival; 

        SelectionOperator(){}

        virtual ~SelectionOperator(){}
        
        virtual vector<size_t> select(const MatrixXd& F, const Parameters& p, const Rnd& r)
        {
            std::cerr << "Error. No selection implementation for base SelectionOperator.\n";
        }
    };
	
	/*!
     * @class Tournament
     */
    struct Tournament: SelectionOperator
    {
        /*!
         * tournament selection operator.
         */
    };
	
	
}
