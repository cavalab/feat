/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef POPULATION_H
#define POPULATION_H

//#include "node.h" // including node.h since definition of node is in the header
using std::vector;
using std::string;
using Eigen::Map;

namespace FT{    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    // individual programs
    struct Individual{
        //Represents individual programs in the population 
        
        vector<Node> program;       // executable data structure
        double fitness;             // aggregate fitness score
        unsigned int loc;           // index of individual in semantic matrix F
        string eqn;                 // symbolic representation of program
        vector<double> weights;     // weights from ML training on program output

        Individual(){}

        ~Individual(){}

        // calculate program output matrix Phi
        MatrixXd out(const MatrixXd& X, const VectorXd& y, const Parameters& params);

        // return symbolic representation of program
        string get_eqn(const Parameters& params);
        
    };

    // population of individuals
    struct Population
    {
        vector<Individual> individuals;

        Population(){}
        ~Population(){}
        
        // initialize population of programs. 
        void init(const Parameters& params);
        
        // reduce programs to the indices in survivors.
        void update(vector<size_t> survivors);

        // make a program.
        //void make_program(vector<Node>& program, const vector<Node>& functions, 
        //                          const vector<Node>& terminals, int max_d, char otype, 
        //                          const vector<double>& term_weights);

    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void make_program(vector<Node>& program, const vector<Node>& functions, 
                                  const vector<Node>& terminals, int max_d, char otype, 
                                  const vector<double>& term_weights)
    {
        /* recursively builds a program with complete arguments. */
        if (max_d == 0)     // append terminal if program has reached max depth
        {
        }
        else
        {
            // let fs be a subset of functions whose output type matches ntype and with an input    
            // type of float if max_d > 1 (assuming all input data is continouous) 
            
            // append a random choice from fs
            Node chosen = functions[0]; 
            program.push_back(chosen);

            // recurse to fulfill the arity of the chosen function
            for (size_t i = 0; i < chosen.arity_f; ++i)
                make_program(program, functions, terminals, max_d-1, 'f', term_weights);
            for (size_t i = 0; i < chosen.arity_b; ++i)
                make_program(program, functions, terminals, max_d-1, 'b', term_weights);

        }
    }

    // calculate program output matrix
    MatrixXd Individual::out(const MatrixXd& X, const VectorXd& y, 
                                const Parameters& params)
    {
        /* evaluate program output. 
         * Input:
         *      X: n_samples x n_features data
         *      y: target data
         *      params: Fewtwo parameters
         * Output:
         *      Phi: n_samples x n_features transformation
         */

        vector<ArrayXd> stack_f; 
        vector<ArrayXi> stack_b;

        // evaluate each node in program
        for (auto& n : program) 
            n.evaluate(X, y, stack_f, stack_b); 
        
        // convert stack_f to Phi
        int rows = stack_f[0].size();
        int cols = stack_f.size();
        double * p = stack_f[0].data();
        // WIP: need to conditional this on the output type parameter
        Map<MatrixXd> Phi (p, rows, cols);       
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

    void Population::init(const Parameters& params)
    {
        /* create random programs in the population, seeded by initial model weights */

        for (auto& ind : individuals){
            // make a program for each individual
            // pick a max depth for this program
            int max_d = 3; // set randomly from [params.min_depth, params.max_depth]

            make_program(ind.program, params.functions, params.terminals, params.max_d,
                         params.otype, params.term_weights);

            // reverse program so that it is post-fix notation
            std::reverse(ind.program.begin(),ind.program.end());
        }
    }
   
   void Population::update(vector<size_t> survivors)
   {
       /* cull population down to survivor indices.*/
       
      individuals.erase(std::remove_if(inviduals.begin(), individuals.end(), 
                        [&survivor](const Inidividual& ind){ return not_in(survivors,ind)}),
                        individuals.end());

   
   }

}
#endif
