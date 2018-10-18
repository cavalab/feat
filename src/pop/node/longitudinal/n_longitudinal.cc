/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_longitudinal.h"
	
namespace FT{
	
    NodeLongitudinal::NodeLongitudinal(std::string n)
    {
        name = "z_"+trim(n);
        
        zName = n;
            
	    otype = 'z';
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 		
    void NodeLongitudinal::evaluate(const Data& data, Stacks& stack)
    {
        try
        {
            stack.z.push(data.Z.at(zName));
        }
        catch (const std::out_of_range& e) 
        {
            cout << "out of range error on ";
            cout << "stack.z.push(data.Z.at(" << zName << "))\n";
            cout << "data.Z size: " << data.Z.size() << "\n";
            cout << "data.Z keys:\n";
            for (const auto& keys : data.Z)
                cout << keys.first << ",";
            cout << "\n";
        }
    }

    /// Evaluates the node symbolically
    void NodeLongitudinal::eval_eqn(Stacks& stack)
    {
        stack.zs.push(name);
    }
    
    NodeLongitudinal* NodeLongitudinal::clone_impl() const { return new NodeLongitudinal(*this); }

    NodeLongitudinal* NodeLongitudinal::rnd_clone_impl() const { return clone_impl(); }

}
