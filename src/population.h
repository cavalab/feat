/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef POPULATION_H
#define POPULATION_H

#include "node.h" // including node.h since definition of node is in the header
using std::vector;
using std::string;

namespace FT{    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    // individual programs
    struct Individual{
        //Represents individual programs in the population 
        
        vector<Node> program;       // executable data structure
        double fitness;             // aggregate fitness score
        unsigned int loc;           // index of individual in semantic matrix F
        string eqn;                 // symbolic representation of program

        Individual(){}

        ~Individual(){}

        // calculate program output matrix Phi
        MatrixXd out(const MatrixXd& X, const VectorXd& y, const Parameters& p);

        // return symbolic representation of program
        string get_eqn(const Parameters& p);
        
    };

    // population of individuals
    struct Population
    {
        vector<Individual> individuals;

        Population(){}
        ~Population(){}

        void init(const Parameters& params)
        {
            // initialize population of programs. 
        }
        void update(vector<size_t> survivors)
        {
            // reduce programs to the indices in survivors.
        }
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    // calculate program output matrix
    MatrixXd Individual::out(const MatrixXd& X, const VectorXd& y, 
                                const Parameters& p)
    {
        MatrixXd Phi; 
        vector<ArrayXd> stack_f; 
        vector<ArrayXi> stack_b;

        // evaluate each node in program
        for (auto n : program)
        {   
            n.evaluate(X, y, stack_f, stack_b);
        }
        // convert stack_f to Phi
        return Phi;
    }

    // return symbolic representation of program 
    string Individual::get_eqn(const Parameters& params)
    {
        if (eqn.empty())               // calculate eqn if it doesn't exist yet 
        {
            vector<string> stack_f;     // symbolic floating stack
            vector<string> stack_b;     // symbolic boolean stack

            for (auto n : program)
                n.eval_eqn(stack_f,stack_b);

            // tie stack outputs together to return representation
            if (params.otype=='b'){
                for (auto s : stack_b) 
                    eqn += "[" + s + "]";
            }
            else
                for (auto s : stack_f) 
                    eqn += "[" + s + "]";
        }

        return eqn;
    }

}
#endif
