/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

namespace FT{

    /*!
     * @class SelectionOperator
     * @brief base class for selection operators.
     */ 
    struct SelectionOperator 
    {
        bool survival; 
        string name;

        //SelectionOperator(){}

        virtual ~SelectionOperator(){}
        
        virtual vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& p) 
        {   
            HANDLE_ERROR_THROW("Undefined select() operation");
        }
        virtual vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& p)
        {
            HANDLE_ERROR_THROW("Undefined select() operation");
        };

    };
	
	
	
	
}
