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
    extern Rnd r;

    // individual programs
    struct Individual{
        /* individual programs in the population. */
        
        vector<Node> program;       // executable data structure
        double fitness;             // aggregate fitness score
        size_t loc;                 // index of individual in semantic matrix F
        string eqn;                 // symbolic representation of program
        vector<double> weights;     // weights from ML training on program output

        Individual(){}

        ~Individual(){}

        // calculate program output matrix Phi
        MatrixXd out(const MatrixXd& X, const VectorXd& y, const Parameters& params);

        // return symbolic representation of program
        string get_eqn(char otype);

        // setting and getting from individuals vector
        const Node operator [](int i) const {return program[i];}
        const Node & operator [](int i) {return program[i];}

        // overload = to copy just the program
        Individual& operator=(Individual rhs)   // note: pass-by-value for implicit copy of rhs
        {
            std::swap(this->program , rhs.program);
            return *this;            
        }

        // size
        int size(){ return program.size(); }
        
        // grab sub-tree locations given starting point.
        size_t subtree(size_t i, char otype);

       // // get program depth.
       // unsigned int depth();

        // get program dimensionality
        unsigned int dim();

        private:
   //         unsigned int depth;         // program depth
            unsigned int dim;           // program dimensionality
    };

    // population of individuals
    struct Population
    {
        vector<Individual> individuals;

        Population(){}
        Population(int p){individuals.resize(p);}
        ~Population(){}
        
        // initialize population of programs. 
        void init(const Parameters& params);
        
        // reduce programs to the indices in survivors.
        void update(vector<size_t> survivors);
        
        int size(){return individuals.size();}

        // setting and getting from individuals vector
        const Individual operator [](int i) const {return individuals[i];}
        const Individual & operator [](int i) {return individuals[i];}
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
        if (max_d == 0 || r.rnd_flt() < terminals.size()/(terminals.size()+functions.size())) 
        {
            // append terminal 
            vector<size_t> ti, tw;  // indices of valid terminals 
            for (size_t i = 0; i<terminals.size(); ++i)
            {
                if (terminals[i].otype == otype) // grab terminals matching output type
                {
                    ti.push_back(i);
                    tw.push_back(term_weights[i]);
                }
            }
            program.push_back(terminals[r.random_choice(ti,tw)]);
        }
        else
        {
            // let fs be an index of functions whose output type matches ntype and with an input    
            // type of float if max_d > 1 (assuming all input data is continouous) 
            vector<size_t> fi;
            for (size_t i = 0; i<functions.size(); ++i)
            {
                if (functions[i].otype == otype && (max_d>1 || functions[i].arity_b==0))
                    fi.push_back(i);
            }
            // append a random choice from fs            
            program.push_back(functions[r.random_choice(fi)]);
            
            Node chosen = program.back();
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
         *      X: n_features x n_samples data
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
        int cols = stack_f[0].size();
        int rows = stack_f.size();
        double * p = stack_f[0].data();
        // WIP: need to conditional this on the output type parameter
        Map<MatrixXd> Phi (p, rows, cols);       
        return Phi;
    }

    // return symbolic representation of program 
    string Individual::get_eqn(char otype)
    {
        if (eqn.empty())               // calculate eqn if it doesn't exist yet 
        {
            vector<string> stack_f;     // symbolic floating stack
            vector<string> stack_b;     // symbolic boolean stack

            for (auto n : program)
                n.eval_eqn(stack_f,stack_b);

            // tie stack outputs together to return representation
            if (otype=='b'){
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
        std::cout << "population size: " << individuals.size() << "\n";
        size_t count = 0;
        for (auto& ind : individuals){
            // make a program for each individual
            // pick a max depth for this program
            // pick a dimensionality for this individual
            int dim = r.rnd_int(1,params.max_dim);

            for (unsigned int i = 0; i<dim; ++i)
            {
                // pick depth from [params.min_depth, params.max_depth]
                int depth =  r.rnd_int(1, params.max_depth);
                
                make_program(ind.program, params.functions, params.terminals, depth,
                             params.otype, params.term_weights);               
                
            }
            // reverse program so that it is post-fix notation
            std::reverse(ind.program.begin(),ind.program.end());
            std::cout << ind.get_eqn(params.otype) << "\n"; // test output
            
            // set location of individual and increment counter
            ind.loc = count;         
            ++count;               
        }
    }
   
   void Population::update(vector<size_t> survivors)
   {
       /* cull population down to survivor indices.*/
       
      individuals.erase(std::remove_if(individuals.begin(), individuals.end(), 
                        [&survivors](const Individual& ind){ return not_in(survivors,ind.loc);}),
                        individuals.end());

      // reset individual locations and F matrix to match

   
   }

   size_t Individual::subtree(size_t i, char otype='0')
   {

       /* finds indices of subtree in program with root i.
        * Input: i, root index of subtree
        * Output: k, last index in subtree
        * note that this function assumes a subtree's arguments to be contiguous in the program.
        */
        
       size_t i2 = i;                              // index for second recursion

       if (program[i].otype == otype || otype=='0')     // if this node is a subtree argument
       {
           for (unsigned int j = 0; j<program[i].arity_f; ++j)
               subtree(--i,'f');                  // recurse for floating arguments
           for (unsigned int j = 0; j<program[i2].arity_b; ++j)
               subtree(--i2,'b');                 // recurse for boolean arguments
       }
       return std::min(i,i2);
   }
   
   //// get program depth.
   //unsigned int Individual::depth()
   //{
   //    /* returns the maximum depth of program.
   //     * the depth is calculated by looping thru the program and incrementing whenever the 
   //     * arity increases, and resetting whenever it is zero.
   //     */
   //    if (depth == 0)      // only calculate if depth hasn't been assigned
   //    {
   //        unsigned int tmp_depth = 0;
   //        unsigned int ca=0, pa=0;     // current arity, previous arity
   //        for (unsigned int i = program.size()-1; i>=0; --i)
   //        {
   //            pa = ca;
   //            ca += program[i].total_arity() - 1;

   //            if (ca > pa)
   //                ++tmp_depth;
   //            else if (ca == 0) 
   //            {
   //                if (tmp_depth > depth)
   //                    depth = tmp_depth;
   //                tmp_depth = 0;
   //            }                 
   //        }
   //    }
   //}

   // get program dimensionality
   unsigned int dim()
   {
       /* returns the dimensionality, i.e. number of outputs, of a program.
       *  the dimensionality is equal to the number of times the program arities are fully
       *  satisfied. 
       */
       if (dim == 0)        // only calculate if dim hasn't been assigned
       {
           unsigned int ca=0;     // current arity
           for (unsigned int i = program.size()-1; i>=0; --i)
           {
               ca += program[i].total_arity() - 1;
               if (ca == 0) ++dim;

           }
       }
       return dim;
   }

}
#endif
