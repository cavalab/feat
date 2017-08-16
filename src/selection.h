/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef SELECTION_H
#define SELECTION_H
//#include <vector>
//using std::vector;
using std::cerr;

namespace FT{
    struct Parameters; // forward declaration of Parameters       
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    struct SelectionOperator 
    {
        // base class for selection operators. 
        bool survival; 

        SelectionOperator();
        virtual ~SelectionOperator();
        virtual vector<size_t> select(const MatrixXd& F, const Parameters& p)
        {
            std::cerr << "Error. No selection implementation for base SelectionOperator.\n";
        };
    };

    struct Lexicase : SelectionOperator
    {
        // Lexicase selection operator.
        Lexicase(bool surv){ survival = surv; };

        vector<size_t> select(const MatrixXd& F, const Parameters& p)
        {
            cout << "Lexicase selection\n";
        };

    };

    struct Tournament: SelectionOperator
    {
        // tournament selection operator.
    };

    struct Pareto : SelectionOperator
    {
        // Pareto selection operator.
        Pareto(bool surv){ survival = surv; };

    };
    
    struct Selection
    {
        // implements selection methods. 
        SelectionOperator *pselector; 
        
        Selection(string type, bool survival=false)
        {
            // set type of selection operator. 
            if (!type.compare("lexicase"))
                pselector = new Lexicase(survival); 
            else if (!type.compare("pareto"))
                pselector = new Pareto(survival);
                
        };
        ~Selection(){
            delete pselector; 
        };
        
        // perform selection
        vector<size_t> select(const MatrixXd& F, const Parameters& p)
        {       
            return pselector->select(F, p);
        };
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
}
#endif
