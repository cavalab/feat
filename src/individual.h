/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "stack.h"
#ifdef USE_CUDA
    #include "node-cuda/cuda_utils.h"
#endif

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class Individual
     * @brief individual programs in the population
     */
    class Individual{
    public:        
        NodeVector program;                            ///< executable data structure
        double fitness;             				///< aggregate fitness score
        size_t loc;                 				///< index of individual in semantic matrix F
        string eqn;                 				///< symbolic representation of program
        vector<double> w;            				///< weights from ML training on program output
        vector<double> p;                           ///< probability of variation of subprograms
        unsigned int dim;           				///< dimensionality of individual
        vector<double> obj;                         ///< objectives for use with Pareto selection
        unsigned int dcounter;                      ///< number of individuals this dominates
        vector<unsigned int> dominated;             ///< individual indices this dominates
        unsigned int rank;                          ///< pareto front rank
        float crowd_dist;                           ///< crowding distance on the Pareto front
        
        
        Individual(){c = 0; dim = 0; eqn="";}

        /// calculate program output matrix Phi
        MatrixXd out(const MatrixXd& X, 
                     const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                     const Parameters& params,
                     const VectorXd& y);

        /// return symbolic representation of program
        string get_eqn();
        
        /// return program name list 
        string program_str() const;

        /// setting and getting from individuals vector
        /* const std::unique_ptr<Node> operator [](int i) const {return program.at(i);} */ 
        /* const std::unique_ptr<Node> & operator [](int i) {return program.at(i);} */

        /// set rank
        void set_rank(unsigned r){rank=r;}
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
        
        ///// get weighted probabilities
        //vector<double> get_w(){ return w;}
        ///// get weight probability for program location i 
        //double get_w(const size_t i);
        ///// set weighted probabilities
        //void set_w(vector<double>& weights);

        /// make a deep copy of the underlying program 
        /* void program_copy(vector<std::unique_ptr<Node>>& cpy) const */
        /* { */
        /*     cpy.clear(); */
        /*     for (const auto& p : program) */
        /*         cpy.push_back(p->clone()); */
        /* } */
        /// clone this individual 
        void clone(Individual& cpy)
        {
            cpy.program = program;
            cpy.p = p;
        }
        /// get probabilities of variation
        vector<double> get_p(){ return p; }     
        /// get inverted weight probability for pogram location i
        double get_p(const size_t i);
        /// get probability of variation for program locations locs
        vector<double> get_p(const vector<size_t>& locs); 
        /// get maximum stack size needed for evaluation.
        std::map<char,size_t> get_max_stack_size();

        unsigned int c;            ///< the complexity of the program.    
        vector<char> dtypes;       ///< the data types of each column of the program output
    
        /// set probabilities
        void set_p(const vector<double>& weights, const double& fb);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions

    //void Individual::set_w(vector<double>& weights)
    //{
    //    // w should have an element corresponding to each root node. 
    //    if (roots().size() != weights.size())
    //        std::cout << "roots size: " << roots().size() << ", w size: " << w.size() << "\n";
    //    
    //    assert(roots().size() == weights.size());
    //    w = weights; 
    //    set_p();
    //}
    //double Individual::get_w(const size_t i)
    //{
    //    /*! @param i index in program 
    //     * @returns weight associated with node */
    //    vector<size_t> rts = roots();
    //    
    //    size_t j = 0;
    //    double size = rts[0];
    //    while ( j < rts.size())
    //    {
    //        if (j > 1) 
    //            size = rts[j] - rts[j-1];
    //        if (i == rts[j])
    //            return w.at(j)/size;    
    //        else if (i > rts[j])
    //            ++j;
    //        
    //    }
    //    // normalize weight by size of subtree
    //    double norm_weight = w.at(j)/size;
    //    return norm_weight;
    //}
    void Individual::set_p(const vector<double>& weights, const double& fb)
    {   
        //cout<<"Weights size = "<<weights.size()<<"\n";
        //cout<<"Roots size = "<<roots().size()<<"\n";
        if(weights.size() != roots().size())
        {
            cout<<"Weights are\n";
            for(double weight : weights)
                cout<<weight<<"\n";
                
            cout<<"Roots are\n";
            auto root1 = roots();
            for(auto root : root1)
                cout<<root<<"\n";
            
            cout<<"Program is \n";
            for (const auto& p : program) std::cout << p->name << " ";
            cout<<"\n";
                
        }
        assert(weights.size() == roots().size());     
        p = weights;
        for (unsigned i=0; i<p.size(); ++i)
            p[i] = 1-p[i];
        double u = 1.0/double(p.size());    // uniform probability
        p = softmax(p);
        for (unsigned i=0; i<p.size(); ++i)
            p[i] = u + fb*(u-p[i]);
    }
    double Individual::get_p(const size_t i)
    {
        /*! @param i index in program 
         * @returns weight associated with node */
        vector<size_t> rts = roots();
        std::reverse(rts.begin(),rts.end()); 
        size_t j = 0;
        double size = rts[0];
        
        while ( j < rts.size())
        {
            if (j > 1) 
                size = rts.at(j) - rts.at(j-1);
            
            if (i <= rts.at(j))
                return p.at(j)/size;    
            else
                ++j;
        }
        
        // normalize weight by size of subtree
        double norm_weight = p.at(j)/size;
        return norm_weight;

    }
    vector<double> Individual::get_p(const vector<size_t>& locs)
    {
        vector<double> ps;
        for (const auto& el : locs) ps.push_back(get_p(el));
        return ps;
    }
#ifndef USE_CUDA
    // calculate program output matrix
    MatrixXd Individual::out(const MatrixXd& X,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                             const Parameters& params, 
                             const VectorXd& y = VectorXd())
    {
        /*!
         * @params X: n_features x n_samples data
         * @params Z: longitudinal nodes for samples
         * @params y: target data
         * @params: Feat parameters
         * @returns Phi: n_features x n_samples transformation
         */

        Stacks stack;
        params.msg("evaluating program " + get_eqn(),2);
        params.msg("program length: " + std::to_string(program.size()),2);
        // evaluate each node in program
        for (const auto& n : program)
        {
        	if(stack.check(n->arity))
        	{
        	    //cout<<"***enter here "<<n->name<<"\n";
	            n->evaluate(X, y, Z, stack);
	            //cout<<"***exit here "<<n->name<<"\n";
	        }
            else
            {
                std::cout << "individual::out() error: node " << n->name << " in " + program_str() + 
                             " is invalid\n";
                std::cout << "float stack size: " << stack_f.size() << "\n";
                std::cout << "bool stack size: " << stack_b.size() << "\n";
                std::cout << "op arity: " << n->arity['f'] << "f, " << n->arity['b'] << "b\n";
                exit(1);
            }
        }
        /* std::cout << "\n"; */
        // convert stack_f to Phi
        params.msg("converting stacks to Phi",2);
        int cols;
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
            {   std::cout << "Error: no outputs in stacks\n"; throw;}
            
            cols = stack.b.top().size();
        }
        else
            cols = stack.f.top().size();
               
        int rows_f = stack.f.size();
        int rows_b = stack.b.size();
        
        dtypes.clear();        
        Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_b, cols);
        // add stack_f to Phi
        for (unsigned int i=0; i<rows_f; ++i)
        {    Phi.row(i) = VectorXd::Map(stack.f.at(i).data(),cols);
             dtypes.push_back('f'); 
        }
        // convert stack_b to Phi       
        for (unsigned int i=0; i<rows_b; ++i)
        {
            Phi.row(i+rows_f) = ArrayXb::Map(stack.b.at(i).data(),cols).cast<double>();
            dtypes.push_back('b');
        }       
        //Phi.transposeInPlace();
        return Phi;
    }
#else //////////////////////////////////////////////////////////// GPU implementation
    // calculate program output matrix on GPU
    MatrixXd Individual::out(const MatrixXd& X,
                             const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z,
                             const Parameters& params, 
                             const VectorXd& y = VectorXd())
    {
        /*!
         * @params X: n_features x n_samples data
         * @params Z: longitudinal nodes for samples
         * @params y: target data
         * @params: Feat parameters
         * @returns Phi: n_features x n_samples transformation
         */

        Stacks stack;
        params.msg("evaluating program " + get_eqn(),2);
        params.msg("program length: " + std::to_string(program.size()),2);
        // to minimize copying overhead, set the stack size to the maximum it will reach for the
        // program 
        std::map<char, size_t> stack_size = get_max_stack_size(); 
        // set the device based on the thread number
        choose_gpu();        
        
        // allocate memory for the stack on the device
        std::cout << "X size: " << X.rows() << "x" << X.cols() << "\n"; 
        stack.allocate(stack_size,X.cols());        
        /* stack.f.resize( */
        // evaluate each node in program
        for (const auto& n : program)
        {
        	if(stack.check(n->arity))
        	{
        	    //cout<<"***enter here "<<n->name<<"\n";
	            n->evaluate(X, y, Z, stack);
	            //cout<<"***exit here "<<n->name<<"\n";
                // adjust indices
                stack.update_idx(n->otype, n->arity); 
	        }
            else
            {
                std::cout << "individual::out() error: node " << n->name << " in " + program_str() + 
                             " is invalid\n";
                std::cout << "float stack size: " << stack.f.size() << "\n";
                std::cout << "bool stack size: " << stack.b.size() << "\n";
                std::cout << "op arity: " << n->arity['f'] << "f, " << n->arity['b'] << "b\n";
                exit(1);
            }
        }
        // copy data from GPU to stack
        stack.copy_to_host(stack_size);
        // remove extraneous rows from stacks
        stack.trim();
        //check stack
        std::cout << "stack.f:" << stack.f.rows() << "x" << stack.f.cols() << "\n";
        for (unsigned i = 0; i < stack.f.rows() ; ++i){
            for (unsigned j = 0; j<10 ; ++j)
                std::cout << stack.f(i,j) << ",";
            std::cout << "\n\n";
        }
        std::cout << "stack.b:" << stack.b.rows() << "x" << stack.b.cols() << "\n";
        for (unsigned i = 0; i < stack.b.rows() ; ++i){
            for (unsigned j = 0; j<10 ; ++j)
                std::cout << stack.b(i,j) << ",";
            std::cout << "\n\n";
        }
        // convert stack_f to Phi
        params.msg("converting stacks to Phi",2);
        int cols;
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
            {   std::cout << "Error: no outputs in stacks\n"; throw;}
            
            cols = stack.b.cols();
        }
        else
            cols = stack.f.cols();
               
        int rows_f = stack.f.rows();
        int rows_b = stack.b.rows();
        
        dtypes.clear();        
        Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_b, cols);
        
        ArrayXXb  PhiB = ArrayXXb::Map(stack.b.data(),stack.b.rows(),stack.b.cols());
        ArrayXXf PhiF = ArrayXXf::Map(stack.f.data(),stack.f.rows(),stack.f.cols());
        // combine stacks into Phi 
        Phi <<  PhiF.cast<double>(),
                PhiB.cast<double>();
        
        std::cout << "Phi:" << Phi.rows() << "x" << Phi.cols() << "\n";

        for (unsigned int i=0; i<rows_f; ++i)
        {    
             /* Phi.row(i) = VectorXd::Map(stack.f.at(i).data(),cols); */
             dtypes.push_back('f'); 
        }
        // convert stack_b to Phi       
        for (unsigned int i=0; i<rows_b; ++i)
        {
            /* Phi.row(i+rows_f) = ArrayXb::Map(stack.b.at(i).data(),cols).cast<double>(); */
            dtypes.push_back('b');
        }
               
        //Phi.transposeInPlace();
        return Phi;
    }
#endif

    // return symbolic representation of program 
    string Individual::get_eqn()
    {
        if (eqn.empty())               // calculate eqn if it doesn't exist yet 
        {
            Stacks stack;

            for (const auto& n : program){
            	if(stack.check_s(n->arity))
                	n->eval_eqn(stack);
                else
                {
                    std::cout << "get_eqn() error: node " << n->name 
                              << " in " + program_str() + " is invalid\n";
                    exit(1);
                }

            }
            // tie stack outputs together to return representation
            for (auto s : stack.fs) 
                eqn += "[" + s + "]";
            for (auto s : stack.bs) 
                eqn += "[" + s + "]";              
            for (auto s : stack.zs) 
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
       
       size_t tmp = i;
       assert(i>=0 && "attempting to grab subtree with index < 0");
              
       if (program[i]->total_arity()==0)    // return this index if it is a terminal
           return i;
       
       std::map<char, unsigned int> arity = program[i]->arity;

       if (otype!='0')  // if we are recursing (otype!='0'), we need to find 
                        // where the nodes to recurse are.  
       {
           while (i>0 && program[i]->otype != otype) --i;    
           assert(program[i]->otype == otype && "invalid subtree arguments");
       }
              
       for (unsigned int j = 0; j<arity['f']; ++j)  
           i = subtree(--i,'f');                   // recurse for floating arguments      
       size_t i2 = i;                              // index for second recursion
       for (unsigned int j = 0; j<arity['b']; ++j)
           i2 = subtree(--i2,'b');
       size_t i3 = i2;                 // recurse for boolean arguments
       for (unsigned int j = 0; j<arity['z']; ++j)
           i3 = subtree(--i3,'z'); 
       return std::min(i,i3);
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
        obj.clear();
        
        for (const auto& n : objectives)
        {
            if (n.compare("fitness")==0)
                obj.push_back(fitness);
            else if (n.compare("complexity")==0)
                obj.push_back(complexity());
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
        // find "root" nodes of program, where roots are final values that output 
        // something directly to the stack
        // assumes a program's subtrees to be contiguous
         
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
           
            total_arity += program[i-1]->total_arity(); 
           
        }
       
        return indices; 
    }

    string Individual::program_str() const
    {
        /* returns a string of program names. */
        string s = "";
        for (const auto& p : program)
        {
            s+= p->name;
            s+=" ";
        }
        return s;
    }

    std::map<char, size_t> Individual::get_max_stack_size()
    {
        // max stack size is calculated using node arities
        std::map<char, size_t> current_stack_size;
        current_stack_size['f']=0; current_stack_size['b']=0;
        std::map<char, size_t> max_stack_size; 
        max_stack_size['f']=0; max_stack_size['b']=0;
        for (const auto& n : program)   
        {   
            ++current_stack_size[n->otype];
            
            if (current_stack_size[n->otype] > max_stack_size[n->otype])
                max_stack_size[n->otype] = current_stack_size[n->otype]; 
            
            for (const auto& a : n->arity)
                current_stack_size[a.first] -= a.second;
                       
        }
        std::cout << "stack size: \n";
        for (const auto& s : max_stack_size)
            std::cout << s.first << ":" << s.second << "\n";
        return max_stack_size;
    }
}

#endif
