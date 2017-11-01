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
        vector<unsigned int> dominated;             ///< individual indices this dominates
        unsigned int rank;                          ///< pareto front rank
        float crowd_dist;                           ///< crowding distance on the Pareto front
        
        Individual(){c = 0; dim = 0;}

        ~Individual(){}

        /// calculate program output matrix Phi
        MatrixXd out(const MatrixXd& X, const Parameters& params, const VectorXd& y);

        /// return symbolic representation of program
        string get_eqn(char otype);
        
        /// setting and getting from individuals vector
        const std::shared_ptr<Node> operator [](int i) const {return program[i];}
        const std::shared_ptr<Node> & operator [](int i) {return program[i];}

        /// overload = to copy just the program
        Individual& operator=(Individual rhs)   // note: pass-by-value for implicit copy of rhs
        {
            std::swap(this->program , rhs.program);
            return *this;            
        }

        /// return size of program
        int size() const { return program.size(); }
        
        /// grab sub-tree locations given starting point.
        size_t subtree(size_t i, char otype) const;

       // // get program depth.
       // unsigned int depth();

        /// get program dimensionality
        unsigned int get_dim();
        
        /// check whether this dominates b. 
        int check_dominance(const Individual& b) const;
        
        /// set obj vector given a string of objective names
        void set_obj(const vector<string>&); 
        
        /// calculate program complexity. 
        unsigned int complexity();
      
        /// find root locations in program.
        vector<size_t> roots();
       
        unsigned int c;            ///< the complexity of the program.    
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions

    // calculate program output matrix
    MatrixXd Individual::out(const MatrixXd& X, const Parameters& params, 
                             const VectorXd& y = VectorXd())
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
        Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows, cols);
        
        for (unsigned int i=0; i<rows; ++i)
            Phi.row(i) = VectorXd::Map(stack_f[i].data(),cols);
        ////check Phi
        //for (int i =0; i<rows; ++i){
        //    std::cout <<"stack_f["<< i <<"]: " << stack_f[i].transpose() << "\n";
        //}
        //for (int i=0; i<rows; ++i){
        //    std::cout <<"  Phi("<<i<<",:): " << Phi.row(i) << "\n";  
        //} 

        //Phi.transposeInPlace();
        return Phi;
    }

    // return symbolic representation of program 
    string Individual::get_eqn(char otype='f')
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
    
    size_t Individual::subtree(size_t i, char otype='0') const 
    {

       /*!
        * finds index of the end of subtree in program with root i.
        
        * Input:
        
        *		i, root index of subtree
        
        * Output:
        
        *		last index in subtree, <= i
        
        * note that this function assumes a subtree's arguments to be contiguous in the program.
        */
        
       if (program[i]->total_arity()==0) return i;

       std::map<char, unsigned int> arity = program[i]->arity;

       if (otype!='0')  // otype=0 means we are at the root. if we are recursing, we need to find 
                        // where the nodes to recurse are.  
       {
           while (program[i-1]->otype != otype && i > 0) --i;
           assert(program[i]->otype == otype && "invalid subtree arguments");
       }
       size_t i2 = i;                              // index for second recursion
       
       for (unsigned int j = 0; j<arity['f']; ++j)
           i = subtree(--i,'f');                  // recurse for floating arguments
       for (unsigned int j = 0; j<arity['b']; ++j)
           i2 = subtree(--i2,'b');                 // recurse for boolean arguments
      
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
            
            for (unsigned int i = program.size(); i>0; --i)
            {
                ca += program[i-1]->total_arity();
                if (ca == 0) ++dim;
                else --ca;
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
         *      1: this individual dominates b; -1: b dominates this; 0: neither dominates
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
                    obj.push_back(complexity());
            }
        }
    }

    unsigned int Individual::complexity()
    {
        if (c==0)
        {
            std::map<char, vector<unsigned int>> stack_c; 
            
            for (const auto& n : program)
                n->eval_complexity(stack_c);
        
            for (const auto& s : stack_c)
                for (const auto& t : s.second)
                    c += t;
            //// debug
            //std::map<char, vector<string>> stack_cs; 
            //string complex_eqn;
            //for (const auto& n : program)
            //    n->eval_complexity_db(stack_cs);
            //
            //for (const auto& s : stack_cs)
            //    for (const auto& t : s.second)
            //        complex_eqn += "+" + t;

            //std::cout << "eqn: " + eqn + ", complexity: " + complex_eqn +"=" +std::to_string(c) + "\n";
        }
        return c;
    }

    vector<size_t> Individual::roots()
    {
        // find "root" nodes of floating point program, where roots are final values that output 
        // something directly to the stack
        vector<size_t> indices;     // returned root indices
        int total_arity = -1;       //end node is always a root
        
        for (size_t i = program.size(); i>0; --i)   // reverse loop thru program
        {            
            if (total_arity <= 0 ){ // root node
                indices.push_back(i-1);
                total_arity=0;
            }
            else
                --total_arity;
           
            total_arity += program[i]->total_arity(); 
           
        }    
        return indices;
    }
}

#endif
