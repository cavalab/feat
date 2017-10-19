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
        vector<double> obj;                         ///< objectives for use with Pareto selection
        unsigned int dcounter;                      ///< number of individuals this dominates
        vector<unsigned int> dominated              ///< individual indices this dominates
        unsigned int rank;                          ///< pareto front rank
        float crowd_dist;                           ///< crowding distance on the Pareto front
        
        Individual(){c = 0;}

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
        /// check whether this dominates b. 
        int check_dominance(const Individual& b) const;
        /// set obj vector given a string of objective names
        void set_obj(const vector<string>&); 
        /// calculate program complexity. 
        unsigned int complexity();

        private:
            unsigned int c;            ///< the complexity of the program.    
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
        vector<ArrayXb> stack_b;

        // evaluate each node in program
        for (auto n : program)
        {
        	if(stack_f.size() >= n->arity['f'] && stack_b.size() >= n->arity['b'])
	            n->evaluate(X, y, stack_f, stack_b); 
        }
        
        // convert stack_f to Phi
        int cols = stack_f[0].size();
        int rows = stack_f.size();
        double * p = stack_f[0].data();
        // TODO: need to conditional this on the output type parameter
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
            	if(stack_f.size() >= n->arity['f'] && stack_b.size() >= n->arity['b'])
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

    int Individual::check_dominance(const Individual& b) const
    {
        /* Check whether this individual dominates b. 
         *
         * Input:
         *
         *      b: another individual
         *
         * Output:
         *
         *      1: this individual domintes b; -1: b dominates this; 0: neither dominates
         */

        int flag1 = 0, // to check if this has a smaller objective
            flag2 = 0; // to check if b    has a smaller objective

        for (int i=0; i<obj.size(); ++i) {
            if (obj[i] < b.obj[i]) 
                flag1 = 1;
            else if (obj[i] > b.obj[i]) 
                flag2 = 1;                        
        }

        if (flag1==1 && flag2==0)   
            // there is at least one smaller objective for this and none for b
            return 1;               
        else if (flag1==0 && flag2==1) 
            // there is at least one smaller objective for b and none for this
            return -1;
        else             
            // no smaller objective or both have one smaller
            return 0;

    }

    void Individual::set_obj(const vector<string>& objectives)
    {
        /*! Input:
         *      objectives: vector of strings naming objectives.
         */
        if (obj.empty())
        {
            for (const auto& n : objectives)
            {
                if (n.compare("fitness")==0)
                    obj.push_back(fitness);
                else if (n.compare("complexity")==0)
                    obj.push_back(c);
            }
        }
    }

    unsigned int Individual::complexity()
    {
        if (c==0)
        {
            vector<unsigned int> stack_c; 

            for (const auto& n : program)
                n.eval_complexity(stack_c);
        
            c = stack_c.back();
        }
        return complexity;
    }
}

#endif
