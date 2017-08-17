/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H
//#include <vector>
//using std::vector;


namespace FT{
    struct Parameters; // forward declaration of Parameters       
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    struct SelectionOperator 
    {
        /* base class for selection operators. */

        bool survival; 

        SelectionOperator(){}

        virtual ~SelectionOperator(){}
        
        virtual vector<size_t> select(const MatrixXd& F, const Parameters& p)
        {
            std::cerr << "Error. No selection implementation for base SelectionOperator.\n";
        }
    };

    struct Lexicase : SelectionOperator
    {
        /* Lexicase selection operator. */

        Lexicase(bool surv){ survival = surv; }
        
        ~Lexicase(){}

        // select function returns a set of selected indices from F. 
        vector<size_t> select(const MatrixXd& F, const Parameters& p);

    };

    struct Tournament: SelectionOperator
    {
        /* tournament selection operator. */
    };

    struct Pareto : SelectionOperator
    {
        /* Pareto selection operator. */
        Pareto(bool surv){ survival = surv; };
        
        ~Pareto(){}
    };
    
    struct Selection
    {
        // implements selection methods. 
        shared_ptr<SelectionOperator> pselector; 
        
        Selection(string type="lexicase", bool survival=false)
        {
            /* set type of selection operator. */

            if (!type.compare("lexicase"))
                pselector = std::make_shared<Lexicase>(survival); 
            else if (!type.compare("pareto"))
                pselector = std::make_shared<Pareto>(survival);
            else
                std::cerr << "Undefined Selection Operator" + type + "\n";
                
        };

        ~Selection(){}
        
        // perform selection by pointing to the select command for the SelectionOperator
        vector<size_t> select(const MatrixXd& F, const Parameters& p)
        {       
            return pselector->select(F, p);
        }
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    vector<size_t> Lexicase::select(const MatrixXd& F, const Parameters& p)
    {
        /* conducts lexicase selection using semantic matrix F. 
         * Inputs:
         *      F: samples x pop_size matrix defining the raw population errors, i.e. its semantics
         *      params: Fewtwo parameters
         * Outputs:
         *      choices: vector of indices corresponding to selected individuals and cols in F
         */
    }
}
#endif
