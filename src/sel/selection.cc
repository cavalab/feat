/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "selection.h"

namespace FT{
namespace Sel{
       
Selection::Selection(string type, bool survival)
{
    /*!
     * set type of selection operator.
     */
    this->type = type;
    this->survival = survival;
    this->set_operator();
}

void Selection::set_operator()
{
    if (this->type == "lexicase")
        pselector = std::make_shared<Lexicase>(survival); 
    else if (this->type == "fair_lexicase")
        pselector = std::make_shared<FairLexicase>(survival);
    else if (this->type == "nsga2")
        pselector = std::make_shared<NSGA2>(survival);
    else if (this->type == "tournament")
        pselector = std::make_shared<Tournament>(survival);
    else if (this->type == "offspring")    // offspring survival
        pselector = std::make_shared<Offspring>(survival);
    else if (this->type == "random")    // offspring survival
        pselector = std::make_shared<Random>(survival);
    else if (this->type == "simanneal")    // offspring survival
        pselector = std::make_shared<SimAnneal>(survival);
    else
        WARN("Undefined Selection Operator " + this->type + "\n");
        
}

Selection::~Selection(){}

/// return type of selectionoperator
string Selection::get_type(){ return pselector->name; }

/// set type of selectionoperator
void Selection::set_type(string in){ type = in; set_operator();}

/// perform selection 
vector<size_t> Selection::select(Population& pop,  
        const Parameters& params, const Data& d)
{       
    return pselector->select(pop, params, d);
}

/// perform survival
vector<size_t> Selection::survive(Population& pop, 
        const Parameters& params, const Data& d)
{       
    return pselector->survive(pop, params, d);
}

}
}
