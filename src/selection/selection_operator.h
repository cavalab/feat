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
            std::cerr << "Undefined select() operation\n";
            throw;
        }
        virtual vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& p)
        {
            std::cerr << "Undefined select() operation\n";
            throw;
        };

    };
	
	
	
	
}
