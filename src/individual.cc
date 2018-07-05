/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "individual.h"

namespace FT{    
       
    Individual::Individual(){c = 0; dim = 0; eqn="";}

    /// set rank
    void Individual::set_rank(unsigned r){rank=r;}
    /// return size of program
    
    int Individual::size() const { return program.size(); }
    
    /// get number of params in program
    int Individual::get_n_params()
    {
        int n_params =0;
        for (unsigned int i =0; i< program.size(); ++i)
        {
            if (program.at(i)->isNodeDx())
            {
                n_params += program.at(i)->arity['f'];
            }
        }
        return n_params;
    }
    
    unsigned int Individual::get_complexity() const {return c;};
  
    /// clone this individual 
    void Individual::clone(Individual& cpy, bool sameid)
    {
        cpy.program = program;
        cpy.p = p;
        if (sameid)
            cpy.id = id;
    }
    
    void Individual::set_id(unsigned i) { id = i; }
    
    void Individual::set_parents(const vector<Individual>& parents)
    {
        parent_id.clear();
        for (const auto& p : parents)
            parent_id.push_back(p.id);
    }
    
    /// get probabilities of variation
    vector<double> Individual::get_p(){ return p; }     
    
    void Individual::set_p(const vector<double>& weights, const double& fb)
    {   
        //cout<<"Weights size = "<<weights.size()<<"\n";
        //cout<<"Roots size = "<<roots().size()<<"\n";
        if(weights.size() != program.roots().size())
        {
            cout<<"Weights are\n";
            for(double weight : weights)
                cout<<weight<<"\n";
                
            cout<<"Roots are\n";
            auto root1 = program.roots();
            for(auto root : root1)
                cout<<root<<"\n";
            
            cout<<"Program is \n";
            for (const auto& p : program) std::cout << p->name << " ";
            cout<<"\n";
                
        }
        assert(weights.size() == program.roots().size());     
        p.resize(0);
        
        // normalize the sum of the weights
        double sum = 0;
        for (unsigned i =0; i<weights.size(); ++i)
            sum += fabs(weights.at(i));

        p.resize(weights.size());
        for (unsigned i=0; i< weights.size(); ++i)
            p[i] = 1 - fabs(weights[i]/sum);
        /* for (unsigned i=0; i<p.size(); ++i) */
        /*     p[i] = 1-p[i]; */
        double u = 1.0/double(p.size());    // uniform probability
        /* std::cout << "p: "; */
        /* for (auto tmp : p) cout << tmp << " " ; cout << "\n"; */
        /* std::cout << "softmax(p)\n"; */
        p = softmax(p);
        for (unsigned i=0; i<p.size(); ++i)
            p[i] = u + fb*(u-p[i]);
        /* cout << "exiting set_p\n"; */
    }
    
    double Individual::get_p(const size_t i)
    {
        /*! @param i index in program 
         * @returns weight associated with node */
        vector<size_t> rts = program.roots();
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
        if (i >= rts.size() || j == rts.size()) 
        {
            cout << "WARN: bad root index attempt in get_p()\n";
            return 0.0;
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
    
        // calculate program output matrix
    MatrixXd Individual::out_trace(Data d,
                     const Parameters& params, vector<Trace>& stack_trace)
    {
        /*!
         * @params X: n_features x n_samples data
         * @params Z: longitudinal nodes for samples
         * @params y: target data
         * @params: Feat parameters
         * @returns Phi: n_features x n_samples transformation
         */

        Stacks stack;
        /* params.msg("evaluating program " + get_eqn(),3); */
        /* params.msg("program length: " + std::to_string(program.size()),3); */

        vector<size_t> roots = program.roots();
        size_t root = 0;
        bool trace=false;
        size_t trace_idx=0;

        if (program.at(roots.at(root))->isNodeDx())
        {
            trace=true;
            stack_trace.push_back(Trace());
        }
        
        // evaluate each node in program
        for (unsigned i = 0; i<program.size(); ++i)
        {
            if (i > roots.at(root)){
                ++root;
                if (program.at(roots.at(root))->isNodeDx())
                {
                    trace=true;
                    stack_trace.push_back(Trace());
                    ++trace_idx;
                }
                else
                    trace=false;
            }
            if(stack.check(program.at(i)->arity))
        	{
                if (trace)
                {
                    /* cout << "storing trace of " << program.at(i)->name << "with " << */
                    /*        program.at(i)->arity['f'] << " arguments\n"; */
                    for (int j = 0; j < program.at(i)->arity['f']; j++) {
                        /* cout << "push back float arg for " << program.at(i)->name << "\n"; */
#ifndef USE_CUDA
                        stack_trace.at(trace_idx).f.push_back(stack.f.at(stack.f.size() - 
                                                         (program.at(i)->arity['f'] - j)));
#else
                       /* ArrayXd ad = ArrayXd::Map(stack.f, (stack.f.size() - 
                                                         (program.at(i)->arity['f'] - j)), stack.f.cols());
                        stack_trace.at(trace_idx).f.push_back(ad);*/
#endif
                    }
                    for (int j = 0; j < program.at(i)->arity['b']; j++) {
                        /* cout << "push back bool arg for " << program.at(i)->name << "\n"; */
#ifndef USE_CUDA
                        stack_trace.at(trace_idx).b.push_back(stack.b.at(stack.b.size() - 
                                                         (program.at(i)->arity['b'] - j)));
#else
                        /*ArrayXb bd = ArrayXb::Map(stack.f.data,(stack.b.size() - 
                                                         (program.at(i)->arity['b'] - j)), stack.b.cols());
                        stack_trace.at(trace_idx).b.push_back(bd);*/
#endif
                    }
                }
        	    //cout<<"***enter here "<<n->name<<"\n";
	            program.at(i)->evaluate(d, stack);
                program.at(i)->visits = 0;
	            //cout<<"***exit here "<<n->name<<"\n";
	        }
            else
                HANDLE_ERROR_THROW("out() error: node " + program.at(i)->name + " in " + program_str() + " is invalid\n");
        }
        
        // convert stack_f to Phi
        params.msg("converting stacks to Phi",3);
        int cols;
        
#ifndef USE_CUDA
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
                HANDLE_ERROR_THROW("Error: no outputs in stacks");
         
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
        {    
             ArrayXd Row = ArrayXd::Map(stack.f.at(i).data(),cols);
             clean(Row); // remove nans, set infs to max and min
             Phi.row(i) = Row;
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
#else
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
                HANDLE_ERROR_THROW("Error: no outputs in stacks");
            
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
        
        /* std::cout << "Phi:" << Phi.rows() << "x" << Phi.cols() << "\n"; */

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
#endif
    }

    

#ifndef USE_CUDA
    // calculate program output matrix
    MatrixXd Individual::out(Data d,
                             const Parameters& params)
    {
        /*!
         * @params X: n_features x n_samples data
         * @params Z: longitudinal nodes for samples
         * @params y: target data
         * @params: Feat parameters
         * @returns Phi: n_features x n_samples transformation
         */

        Stacks stack;
        
        params.msg("evaluating program " + get_eqn(),3);
        params.msg("program length: " + std::to_string(program.size()),3);
        // evaluate each node in program
        for (const auto& n : program)
        {
        	if(stack.check(n->arity))
        	{
        	    //cout<<"***enter here "<<n->name<<"\n";
	            n->evaluate(d, stack);
	            //cout<<"***exit here "<<n->name<<"\n";
	        }
            else
                HANDLE_ERROR_THROW("out() error: node " + n->name + " in " + program_str() + " is invalid\n");
        }
        
        // convert stack_f to Phi
        params.msg("converting stacks to Phi",3);
        int cols;
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
                HANDLE_ERROR_THROW("Error: no outputs in stacks");
            
            cols = stack.b.top().size();
        }
        else
            cols = stack.f.top().size();
               
        int rows_f = stack.f.size();
        int rows_b = stack.b.size();
        /* cout << "rows_f (stack.f.size()): " << rows_f << "\n"; */ 
        /* cout << "rows_b (stack.b.size()): " << rows_b << "\n"; */ 
        dtypes.clear();        
        Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_b, cols);
        // add stack_f to Phi
        /* cout << "cols: " << cols << "\n"; */ 
      
        for (unsigned int i=0; i<rows_f; ++i)
        {    
             ArrayXd Row = ArrayXd::Map(stack.f.at(i).data(),cols);
             clean(Row); // remove nans, set infs to max and min
             Phi.row(i) = Row;
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
    MatrixXd Individual::out(Data d,
                             const Parameters& params)
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
        /* std::cout << "X size: " << X.rows() << "x" << X.cols() << "\n"; */ 
        stack.allocate(stack_size,d.X.cols());        
        /* stack.f.resize( */
        // evaluate each node in program
        for (const auto& n : program)
        {
        	if(stack.check(n->arity))
        	{
        	    //cout<<"***enter here "<<n->name<<"\n";
	            n->evaluate(d, stack);
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
        /* std::cout << "stack.f:" << stack.f.rows() << "x" << stack.f.cols() << "\n"; */
        /* for (unsigned i = 0; i < stack.f.rows() ; ++i){ */
        /*     for (unsigned j = 0; j<10 ; ++j) */
        /*         std::cout << stack.f(i,j) << ","; */
        /*     std::cout << "\n\n"; */
        /* } */
        /* std::cout << "stack.b:" << stack.b.rows() << "x" << stack.b.cols() << "\n"; */
        /* for (unsigned i = 0; i < stack.b.rows() ; ++i){ */
        /*     for (unsigned j = 0; j<10 ; ++j) */
        /*         std::cout << stack.b(i,j) << ","; */
        /*     std::cout << "\n\n"; */
        /* } */
        // convert stack_f to Phi
        params.msg("converting stacks to Phi",2);
        int cols;
        if (stack.f.size()==0)
        {
            if (stack.b.size() == 0)
                HANDLE_ERROR_THROW("Error: no outputs in stacks");
            
            cols = stack.b.cols();
        }
        else
            cols = stack.f.cols();
               
        int rows_f = stack.f.rows();
        int rows_b = stack.b.rows();
        
        dtypes.clear();        
        Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_b, cols);

	typedef Array<bool, Dynamic, Dynamic> ArrayXXb;
	typedef Array<float, Dynamic, Dynamic> ArrayXXf;
        
        ArrayXXb  PhiB = ArrayXXb::Map(stack.b.data(),stack.b.rows(),stack.b.cols());
        ArrayXXf PhiF = ArrayXXf::Map(stack.f.data(),stack.f.rows(),stack.f.cols());
        // combine stacks into Phi 
        Phi <<  PhiF.cast<double>(),
                PhiB.cast<double>();
        
        /* std::cout << "Phi:" << Phi.rows() << "x" << Phi.cols() << "\n"; */

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
                    HANDLE_ERROR_THROW("get_eqn() error: node " + n->name + " in " + program_str() + " is invalid\n");
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
    
    // return vectorized symbolic representation of program 
    vector<string> Individual::get_features()
    {
        vector<string> features;
        Stacks stack;

        for (const auto& n : program){
            if(stack.check_s(n->arity))
                n->eval_eqn(stack);
            else
                HANDLE_ERROR_THROW("get_eqn() error: node " + n->name + " in " + program_str() + " is invalid\n");
        }
        // tie stack outputs together to return representation
        for (auto s : stack.fs) 
            features.push_back(s);
        for (auto s : stack.bs) 
            features.push_back(s);

        return features;
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
        std::map<char, size_t> stack_size;
	stack_size['f'] = 0;
	stack_size['b'] = 0; 
        for (const auto& n : program)   
        {   
            //printf("Node name is %s\n", n->name.c_str());
	    
	    ++stack_size[n->otype];
            /*for (const auto& a : n->arity)
                stack_size[a.first] -= a.second;*/

	    //printf("Floating stack size is %zu\n", stack_size['f']);
	    //printf("Boolean stack size is %zu\n", stack_size['b']);
                       
        }
        return stack_size;
    }

}
