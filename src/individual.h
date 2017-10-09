/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class Individual
     * @brief individual programs in the population
     */
    struct Individual{
        
        vector<std::shared_ptr<Node>> program;      ///< executable data structure
        double fitness;             				///< aggregate fitness score
        size_t loc;                 				///< index of individual in semantic matrix F
        string eqn;                 				///< symbolic representation of program
        vector<double> weights;     				///< weights from ML training on program output
        unsigned int dim;           				///< dimensionality of individual

        Individual(){}

        ~Individual(){}

        /*!
         * @brief calculate program output matrix Phi
         */
        MatrixXd out(const MatrixXd& X, const VectorXd& y, const Parameters& params);

        /*!
         * @brief return symbolic representation of program
         */
        string get_eqn(char otype);

        /*!
         * @brief setting and getting from individuals vector
         */
        const std::shared_ptr<Node> operator [](int i) const {return program[i];}
        const std::shared_ptr<Node> & operator [](int i) {return program[i];}

        /*!
         * @brief overload = to copy just the program
         */
        Individual& operator=(Individual rhs)   // note: pass-by-value for implicit copy of rhs
        {
            std::swap(this->program , rhs.program);
            return *this;            
        }

        /*!
         * @brief return size of program
         */
        int size() const { return program.size(); }
        
        /*!
         * @brief grab sub-tree locations given starting point.
         */
        size_t subtree(size_t i, char otype='0') const;

       // // get program depth.
       // unsigned int depth();

        /*!
         * @brief get program dimensionality
         */
        unsigned int get_dim();

        private:
            
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions

    // calculate program output matrix
    MatrixXd Individual::out(const MatrixXd& X, const VectorXd& y, 
                                const Parameters& params)
    {
        /*!
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
        for (auto n : program) 
            n->evaluate(X, y, stack_f, stack_b); 
        
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
                n->eval_eqn(stack_f,stack_b);

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
    
    size_t Individual::subtree(size_t i, char otype) const 
    {

       /*!
        * finds indices of subtree in program with root i.
        
        * Input:
        
        *		i, root index of subtree
        
        * Output:
        
        *		k, last index in subtree
        
        * note that this function assumes a subtree's arguments to be contiguous in the program.
        */
        
       size_t i2 = i;                              // index for second recursion

       if (program[i]->otype == otype || otype=='0')     // if this node is a subtree argument
       {
           for (unsigned int j = 0; j<program[i]->arity['f']; ++j)
               i = subtree(--i,'f');                  // recurse for floating arguments
           for (unsigned int j = 0; j<program[i2]->arity['b']; ++j)
               i2 = subtree(--i2,'b');                 // recurse for boolean arguments
       }
       return std::min(i,i2);
    }
   
   
    // get program dimensionality
    unsigned int Individual::get_dim()
    {    
        /*!
         * Output:
         
         *	 	returns the dimensionality, i.e. number of outputs, of a program.
         *   	the dimensionality is equal to the number of times the program arities are fully
         *   	satisfied. 
         */
        if (dim == 0)        // only calculate if dim hasn't been assigned
        {
            unsigned int ca=0;     // current arity
            for (unsigned int i = program.size()-1; i>=0; --i)
            {
                ca += program[i]->total_arity() - 1;
                if (ca == 0) ++dim;
            }
        }  
        return dim;   
    }
}

#endif
