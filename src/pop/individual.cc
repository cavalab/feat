/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "individual.h"

namespace FT{   

    namespace Pop{ 
           
        Individual::Individual(){c = 0; dim = 0; eqn=""; parent_id.clear(); 
            parent_id.push_back(-1);}

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
        vector<float> Individual::get_p() const { return p; }     
        
        void Individual::set_p(const vector<float>& weights, const float& fb, 
                               bool softmax_norm)
        {   
            //cout<<"Weights size = "<<weights.size()<<"\n";
            //cout<<"Roots size = "<<roots().size()<<"\n";
            if(weights.size() != program.roots().size())
            {
                cout<<"Weights are\n";
                for(float weight : weights)
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
            float sum = 0;
            for (unsigned i =0; i<weights.size(); ++i)
                sum += fabs(weights.at(i));

            p.resize(weights.size());
            for (unsigned i=0; i< weights.size(); ++i)
                p[i] = 1 - fabs(weights[i]/sum);
            /* for (unsigned i=0; i<p.size(); ++i) */
            /*     p[i] = 1-p[i]; */
            float u = 1.0/float(p.size());    // uniform probability
            /* std::cout << "p: "; */
            /* for (auto tmp : p) cout << tmp << " " ; cout << "\n"; */
            /* std::cout << "softmax(p)\n"; */
            if (softmax_norm)
                p = softmax(p);
            // do partial uniform, partial weighted probability, using feedback 
            // ratio
            for (unsigned i=0; i<p.size(); ++i)
                p[i] = (1-fb)*u + fb*p[i];
            /* cout << "exiting set_p\n"; */
            // set weights
            this->w = weights;
        }
        
        float Individual::get_p(const size_t i, bool normalize) const
        {
            /*! @param i index in program 
             *  @param normalize (true): normalizes the probability by the size
             *  of the subprogram. 
             *   Useful when the total probability over the program nodes 
             *   should sum to 1.
             * @return weight associated with node */
                
            vector<size_t> rts = program.roots();
            size_t j = 0;
            float size = rts.at(0)+1;
            /* cout << "roots: "; */
            /* for (auto root : rts) cout << root << ", "; */
            /* cout << "\n"; */
            /* cout << "size: " << size << "\n"; */
            
            while ( j < rts.size())
            {
                if (j > 1) 
                    size = rts.at(j) - rts.at(j-1);
                
                if (i <= rts.at(j))
                {
                    float tmp = normalize ? p.at(j)/size : p.at(j) ;
                    /* cout << "returning " << tmp << endl; */ 
                    return normalize ? p.at(j)/size : p.at(j) ;    
                }
                else
                    ++j;
            }
            if (i >= rts.size() || j == rts.size()) 
            {
                cout << "WARN: bad root index attempt in get_p()\n";
                return 0.0;
            }
            // normalize weight by size of subtree
            float tmp = normalize ? p.at(j)/size : p.at(j) ;
            /* cout << "returning " << tmp << endl; */ 
            return tmp; 
        }
        
        vector<float> Individual::get_p(const vector<size_t>& locs, 
                bool normalize) const
        {
            /*! @param locs: program indices to return probabilities for. 
             *  @param normalize (false): normalize probabilities by size of 
             *  subprogram
             *  @returns float vector of probabilities
             */
            vector<float> ps;
            for (const auto& el : locs) 
            {
                /* cout << "getting p for " << el << "\n"; */
                ps.push_back(get_p(el,normalize));
            }
            return ps;
        }
        
        shared_ptr<CLabels> Individual::fit(const Data& d, 
                const Parameters& params, bool& pass)
        {
            // calculate program output matrix Phi
            logger.log("Generating output for " + get_eqn(), 3);
            Phi = out(d, params, false);      
            // calculate ML model from Phi
            logger.log("ML training on " + get_eqn(), 3);
            ml = std::make_shared<ML>(params.ml, true, params.classification,
                    params.n_classes);
            
            shared_ptr<CLabels> yh = ml->fit(Phi,d.y,params,pass,dtypes);

            if (pass)
                set_p(ml->get_weights(),params.feedback,params.softmax_norm);
            else
            {   // set weights to zero
                vector<float> w(Phi.rows(), 0);                     
                set_p(w,params.feedback,params.softmax_norm);
            }
            
            this->yhat = ml->labels_to_vector(yh);
            
            return yh;
        }

        shared_ptr<CLabels> Individual::predict(const Data& d, 
                const Parameters& params)
        {
            // calculate program output matrix Phi
            logger.log("Generating output for " + get_eqn(), 3);
            // toggle validation
            MatrixXf Phi_pred = out(d, params, true);           
            // TODO: guarantee this is not changing nodes

            if (Phi_pred.size()==0)
            {
                if (d.X.cols() == 0)
                    HANDLE_ERROR_THROW("The prediction dataset has no data");
                else
                    HANDLE_ERROR_THROW("Phi_pred is empty");
            }
            // calculate ML model from Phi
            logger.log("ML predicting on " + get_eqn(), 3);
            // assumes ML is already trained
            shared_ptr<CLabels> yhat = ml->predict(Phi_pred);
            return yhat;
        }

        ArrayXXf Individual::predict_proba(const Data& d, 
                const Parameters& params)
        {
            // calculate program output matrix Phi
            logger.log("Generating output for " + get_eqn(), 3);
            // toggle validation
            MatrixXf Phi_pred = out(d, params, true);           
            // TODO: guarantee this is not changing nodes

            if (Phi_pred.size()==0)
                HANDLE_ERROR_THROW("Phi_pred must be generated before "
                        "predict() is called\n");
            // calculate ML model from Phi
            logger.log("ML predicting on " + get_eqn(), 3);
            // assumes ML is already trained
            ArrayXXf yhat = ml->predict_proba(Phi_pred);
            return yhat;
        }

        VectorXf Individual::predict_drop(const Data& d, const Parameters& params, int drop_idx)
        {
            // calculate program output matrix Phi
            logger.log("Generating output for " + get_eqn(), 3);
            // toggle validation
            MatrixXf PhiDrop = Phi;           // TODO: guarantee this is not changing nodes
             
            if (Phi.size()==0)
                HANDLE_ERROR_THROW("Phi must be generated before predict_drop() is called\n");
            if (drop_idx >= 0)  // if drop_idx specified, mask that phi output
            {
                if (drop_idx >= PhiDrop.rows())
                    HANDLE_ERROR_THROW("drop_idx ( " + std::to_string(drop_idx) + " > Phi size (" 
                                       + std::to_string(Phi.rows()) + ")\n");
                cout << "dropping row " + std::to_string(drop_idx) + "\n";
                /* PhiDrop.row(drop_idx) = VectorXf::Zero(Phi.cols()); */
                PhiDrop.row(drop_idx).setZero();
            }
            // calculate ML model from Phi
            /* logger.log("ML predicting on " + get_eqn(), 3); */
            // assumes ML is already trained
            VectorXf yh = ml->predict_vector(PhiDrop);
            return yh;
        }

        VectorXf Individual::predict_vector(const Data& d, const Parameters& params)
        {
            return ml->labels_to_vector(this->predict(d,params));
        }
        
        MatrixXf Individual::state_to_phi(State& state)
        {
            // we want to preserve the order of the outputs in the program 
            // in the order of the outputs in Phi. 
            // get root output types
            this->dtypes.clear();
            for (auto r : program.roots())
            {
                this->dtypes.push_back(program.at(r)->otype);
            }
            // convert state_f to Phi
            logger.log("converting State to Phi",3);
            int cols;
            
            if (state.f.size()==0)
            {
                if (state.c.size() == 0)
                {
                    if (state.b.size() == 0)
                        HANDLE_ERROR_THROW("Error: no outputs in State");
                    
                    cols = state.b.top().size();
                }
                else{
                    cols = state.c.top().size();
                }
            }
            else{
                cols = state.f.top().size();
            }
                   
            // define Phi matrix 
            Matrix<float,Dynamic,Dynamic,RowMajor> Phi (
                    state.f.size() + state.c.size() + state.b.size(),  
                    cols);
            ArrayXf Row; 
            std::map<char,int> rows;
            rows['f']=0;
            rows['c']=0;
            rows['b']=0;

            // add rows (features) to Phi with appropriate type casting
            for (int i = 0; i < this->dtypes.size(); ++i)
            {
                char rt = this->dtypes.at(i);

                switch (rt)
                {
                    case 'f':
                    // add state_f to Phi
                        Row = ArrayXf::Map(state.f.at(rows[rt]).data(),cols);
                        break;
                    case 'c':
                    // convert state_c to Phi       
                        Row = ArrayXi::Map(
                            state.c.at(rows[rt]).data(),cols).cast<float>();
                        break;
                    case 'b':
                    // add state_b to Phi
                        Row = ArrayXb::Map(
                            state.b.at(rows[rt]).data(),cols).cast<float>();
                        break;
                    default:
                        HANDLE_ERROR_THROW("Unknown root type");
                }
                // remove nans, set infs to max and min
                clean(Row); 
                Phi.row(i) = Row;
                ++rows[rt];
            }
            return Phi;
        }
        #ifndef USE_CUDA
        // calculate program output matrix
        MatrixXf Individual::out(const Data& d, const Parameters& params, 
                bool predict)
        {
            /*!
             * @param d: Data structure
             * @param params: Feat parameters
             * @param predict: if true, this guarantees nodes like split do not 
             *  get trained
             * @return Phi: n_features x n_samples transformation
             */
             
            State state;
            
            logger.log("evaluating program " + get_eqn(),3);
            logger.log("program length: " + std::to_string(program.size()),3);
            // evaluate each node in program
            for (const auto& n : program)
            {
                // learning nodes are set for fit or predict mode
                if (n->isNodeTrain())                     
                    dynamic_cast<NodeTrain*>(n.get())->train = !predict;
            	if(state.check(n->arity))
	                n->evaluate(d, state);
                else
                    HANDLE_ERROR_THROW("out() error: node " + n->name + " in " 
                            + program_str() + " failed arity check\n");
                
            }
            
            return state_to_phi(state);
        }
        #else
        MatrixXf Individual::out(const Data& d, const Parameters& params, bool predict)
        {
        
            /*!
             * @params X: n_features x n_samples data
             * @params Z: longitudinal nodes for samples
             * @params y: target data
             * @params: Feat parameters
             * @returns Phi: n_features x n_samples transformation
             */

            State state;
            logger.log("evaluating program " + get_eqn(),3);
            logger.log("program length: " + std::to_string(program.size()),3);
            // to minimize copying overhead, set the state size to the maximum it will reach for the
            // program 
            std::map<char, size_t> state_size = get_max_state_size();
            // set the device based on the thread number
            Op::choose_gpu();        
            
            // allocate memory for the state on the device
            /* std::cout << "X size: " << X.rows() << "x" << X.cols() << "\n"; */ 
            state.allocate(state_size,d.X.cols());        
            /* state.f.resize( */
            // evaluate each node in program
            for (const auto& n : program)
            {
                if (n->isNodeTrain()) // learning nodes are set for fit or predict mode
                    dynamic_cast<NodeTrain*>(n.get())->train = !predict;
            	if(state.check(n->arity))
            	{
	                n->evaluate(d, state);
                    // adjust indices
                    state.update_idx(n->otype, n->arity); 
	            }
                else
                {
                    std::cout << "individual::out() error: node " << n->name << " in " + program_str() + 
                                 " is invalid\n";
                    std::cout << "float state size: " << state.f.size() << "\n";
                    std::cout << "bool state size: " << state.b.size() << "\n";
                    std::cout << "op arity: " << n->arity['f'] << "f, " << n->arity['b'] << "b\n";
                    exit(1);
                }
            }
            // copy data from GPU to state (calls trim also)
            state.copy_to_host();
            // remove extraneous rows from states
            //state.trim();
            //check state
            /* std::cout << "state.f:" << state.f.rows() << "x" << state.f.cols() << "\n"; */
            /* for (unsigned i = 0; i < state.f.rows() ; ++i){ */
            /*     for (unsigned j = 0; j<10 ; ++j) */
            /*         std::cout << state.f(i,j) << ","; */
            /*     std::cout << "\n\n"; */
            /* } */
            /* std::cout << "state.b:" << state.b.rows() << "x" << state.b.cols() << "\n"; */
            /* for (unsigned i = 0; i < state.b.rows() ; ++i){ */
            /*     for (unsigned j = 0; j<10 ; ++j) */
            /*         std::cout << state.b(i,j) << ","; */
            /*     std::cout << "\n\n"; */
            /* } */
            // convert state_f to Phi
            logger.log("converting State to Phi",3);
            int cols;
            
            if (state.f.size()==0)
            {
                if (state.c.size() == 0)
                {
                    if (state.b.size() == 0)
                        HANDLE_ERROR_THROW("Error: no outputs in state");
                    
                    cols = state.b.cols();
                }
                else
                    cols = state.c.cols();
            }
            else
                cols = state.f.cols();
                   
            int rows_f = state.f.rows();
            int rows_c = state.c.rows();
            int rows_b = state.b.rows();
            
            dtypes.clear();        
            Matrix<float,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_b+rows_c, cols);

            // combine states into Phi 
            Phi <<  state.f.cast<float>(),
                    state.c.cast<float>(),
                    state.b.cast<float>();
                    
            
            /* std::cout << "Phi:" << Phi.rows() << "x" << Phi.cols() << "\n"; */

            for (unsigned int i=0; i<rows_f; ++i)
            {    
                 /* Phi.row(i) = VectorXf::Map(state.f.at(i).data(),cols); */
                 dtypes.push_back('f'); 
            }
            
            for (unsigned int i=0; i<rows_c; ++i)
            {    
                 /* Phi.row(i) = VectorXf::Map(state.f.at(i).data(),cols); */
                 dtypes.push_back('c'); 
            }
            
            // convert state_b to Phi       
            for (unsigned int i=0; i<rows_b; ++i)
            {
                /* Phi.row(i+rows_f) = ArrayXb::Map(state.b.at(i).data(),cols).cast<float>(); */
                dtypes.push_back('b');
            }
                   
            return Phi;
        }
        #endif

        #ifndef USE_CUDA
        // calculate program output matrix
        
        MatrixXf Individual::out_trace(const Data& d, const Parameters& params, 
                                       vector<Trace>& state_trace)
        {
            /*!
             * @param X: n_features x n_samples data
             * @param Z: longitudinal nodes for samples
             * @param y: target data
             * @param: Feat parameters
             * @return Phi: n_features x n_samples transformation
             */

            State state;
            logger.log("evaluating program " + program_str(),3);

            vector<size_t> roots = program.roots();
            size_t root = 0;
            bool trace=false;
            size_t trace_idx=-1;

            // if first root is a Dx node, start off storing its subprogram
            if (program.at(roots.at(root))->isNodeDx())
            {
                trace=true;
                ++trace_idx;
                state_trace.push_back(Trace());
            }
            
            // evaluate each node in program
            for (unsigned i = 0; i<program.size(); ++i)
            {
                /* cout << "i = " << i << ", root = " 
                 * << roots.at(root) << "\n"; */
                if (i > roots.at(root))
                {
                    trace=false;
                    if (root + 1 < roots.size())
                    {
                        ++root; // move to next root
                        // if new root is a Dx node, start storing its args
                        if (program.at(roots.at(root))->isNodeDx())
                        {
                            trace=true;
                            ++trace_idx;
                            state_trace.push_back(Trace());
                        }
                    }
                }
                if(state.check(program.at(i)->arity))
            	{
                    if (trace)
                        state_trace.at(trace_idx).copy_to_trace(state, 
                                program.at(i)->arity);

	                program.at(i)->evaluate(d, state);
                    program.at(i)->visits = 0;
	            }
                else
                    HANDLE_ERROR_THROW("out() error: node " 
                            + program.at(i)->name + " in " + program_str() 
                            + " is invalid\n");
            }
            
            return state_to_phi(state);

        }

        #else
        // calculate program output matrix
        MatrixXf Individual::out_trace(const Data& d,
                         const Parameters& params, vector<Trace>& state_trace)
        {
            /*!
             * @params X: n_features x n_samples data
             * @params Z: longitudinal nodes for samples
             * @params y: target data
             * @params: Feat parameters
             * @returns Phi: n_features x n_samples transformation
             */

            State state;
            /* logger.log("evaluating program " + get_eqn(),3); */
            /* logger.log("program length: " + std::to_string(program.size()),3); */
            
            std::map<char, size_t> state_size = get_max_state_size();
            // set the device based on the thread number
            choose_gpu();
            // allocate memory for the state on the device
            /* std::cout << "X size: " << X.rows() << "x" << X.cols() << "\n"; */ 
            state.allocate(state_size,d.X.cols());

            vector<size_t> roots = program.roots();
            size_t root = 0;
            bool trace=false;
            size_t trace_idx=0;

            if (program.at(roots.at(root))->isNodeDx())
            {
                trace=true;
                state_trace.push_back(Trace());
            }
            
            // evaluate each node in program
            for (unsigned i = 0; i<program.size(); ++i)
            {
                if (i > roots.at(root)){
                    ++root;
                    if (program.at(roots.at(root))->isNodeDx())
                    {
                        trace=true;
                        state_trace.push_back(Trace());
                        ++trace_idx;
                    }
                    else
                        trace=false;
                }
                if(state.check(program.at(i)->arity))
            	{
                    if (trace)
                        state_trace.at(trace_idx).copy_to_trace(state, program.at(i)->arity);
                    
                    program.at(i)->evaluate(d, state);
                    state.update_idx(program.at(i)->otype, program.at(i)->arity); 
                    //cout << "\nstack.idx[otype]: " << state.idx[program.at(i)->otype];
                    program.at(i)->visits = 0;
                    //cout << "Evaluated node " << program.at(i)->name << endl;
                    
                }
                else
                    HANDLE_ERROR_THROW("out_trace() error: node " + program.at(i)->name + " in " + program_str() + " is invalid\n");
            }
            
            state.copy_to_host();
            
            // convert state_f to Phi
            logger.log("converting State to Phi",3);
            int cols;
            
            if (state.f.size()==0)
            {
                if (state.c.size() == 0)
                {
                    if (state.b.size() == 0)
                        HANDLE_ERROR_THROW("Error: no outputs in State");
                    
                    cols = state.b.cols();
                }
                else
                    cols = state.c.cols();
            }
            else
                cols = state.f.cols();
                   
            int rows_f = state.f.rows();
            int rows_c = state.c.rows();
            int rows_b = state.b.rows();
            
            dtypes.clear();        
            
            Matrix<float,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_c+rows_b, cols);

            ArrayXXf PhiF = ArrayXXf::Map(state.f.data(),state.f.rows(),state.f.cols());
            ArrayXXi PhiC = ArrayXXi::Map(state.c.data(),state.c.rows(),state.c.cols());
            ArrayXXb PhiB = ArrayXXb::Map(state.b.data(),state.b.rows(),state.b.cols());
            
            // combine State into Phi 
            Phi <<  PhiF.cast<float>(),
                    PhiC.cast<float>(),
                    PhiB.cast<float>();
            
            /* std::cout << "Phi:" << Phi.rows() << "x" << Phi.cols() << "\n"; */

            for (unsigned int i=0; i<rows_f; ++i)
            {    
                 /* Phi.row(i) = VectorXf::Map(state.f.at(i).data(),cols); */
                 dtypes.push_back('f'); 
            }
            
            for (unsigned int i=0; i<rows_c; ++i)
            {    
                 /* Phi.row(i) = VectorXf::Map(state.f.at(i).data(),cols); */
                 dtypes.push_back('c'); 
            }
            
            // convert state_b to Phi       
            for (unsigned int i=0; i<rows_b; ++i)
            {
                /* Phi.row(i+rows_f) = ArrayXb::Map(state.b.at(i).data(),cols).cast<float>(); */
                dtypes.push_back('b');
            }
                   
            return Phi;
        }
        #endif
        
        // return symbolic representation of program 
        
        string Individual::get_eqn()
        {
            #pragma omp critical 
            {
                this->eqn="";
                State state;

                int i = 0;
                for (const auto& n : program)
                {
                    if(state.check_s(n->arity))
                        n->eval_eqn(state);
                    else
                        HANDLE_ERROR_THROW("get_eqn() error: node " 
                                + n->name + " at location " + to_string(i) 
                                + " in [ " + program_str() 
                                + " ] is invalid\n");
                    ++i;
                }
                // tie state outputs together to return representation
                // order by root types
                this->dtypes.clear();
                for (auto r : program.roots())
                {
                    this->dtypes.push_back(program.at(r)->otype);
                }
                std::map<char,int> rows;
                rows['f']=0;
                rows['c']=0;
                rows['b']=0;
                for (int i = 0; i < this->dtypes.size(); ++i)
                {
                    char rt = this->dtypes.at(i);
                    switch (rt)
                    {
                        case 'f':
                            this->eqn += "[" + state.fs.at(rows[rt]) + "]";
                            break;
                        case 'c':
                            this->eqn += "[" + state.cs.at(rows[rt]) + "]";
                            break;
                        case 'b':
                            this->eqn += "[" + state.bs.at(rows[rt]) + "]";
                            break;
                        default:
                            HANDLE_ERROR_THROW("Unknown root type");
                    }
                    ++rows[rt];
                }
            }

            return this->eqn;
        }

        
        // return vectorized symbolic representation of program 
        vector<string> Individual::get_features()
        {
            vector<string> features;
            State state;

            for (const auto& n : program){
                if(state.check_s(n->arity))
                    n->eval_eqn(state);
                else
                    HANDLE_ERROR_THROW("get_eqn() error: node " + n->name + " in " 
                                       + program_str() + " is invalid\n");
            }
            // tie state outputs together to return representation
            for (auto s : state.fs) 
                features.push_back(s);
            for (auto s : state.bs) 
                features.push_back(s);
            for (auto s : state.cs)
                features.push_back(s);

            return features;
        }
       
        // get program dimensionality
        unsigned int Individual::get_dim()
        {    
            /*!
             * Output:
             
             *	 	@return the dimensionality, i.e. number of outputs, of a program.
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
                else if (n.compare("size")==0)
                    obj.push_back(program.size());
                else if (n.compare("CN")==0)    // condition number of Phi
                {
                    CN = condition_number(Phi.transpose());
                    obj.push_back(CN);
                }
                else if (n.compare("corr")==0)    // covariance structure of Phi
                    obj.push_back(mean_square_corrcoef(Phi));
                else if (n.compare("fairness")==0)
                    obj.push_back(fairness);
                else
                    HANDLE_ERROR_THROW(n+" is not a known objective");
            }
        
        }

        unsigned int Individual::complexity()
        {
            if (c==0)
            {
                std::map<char, vector<unsigned int>> state_c; 
                
                for (const auto& n : program)
                    n->eval_complexity(state_c);
            
                for (const auto& s : state_c)
                    for (const auto& t : s.second)
                        c += t;
                //// debug
                //std::map<char, vector<string>> state_cs; 
                //string complex_eqn;
                //for (const auto& n : program)
                //    n->eval_complexity_db(state_cs);
                //
                //for (const auto& s : state_cs)
                //    for (const auto& t : s.second)
                //        complex_eqn += "+" + t;

                //std::cout << "eqn: " + eqn + ", complexity: " + complex_eqn +"=" +std::to_string(c) + "\n";
            }
            return c;
        }

        string Individual::program_str() const
        {
            /* @return a string of node names. */
            string s = "";
            for (const auto& p : program)
            {
                s+= p->name;
                s+=" ";
            }
            return s;
        }
        
        std::map<char, size_t> Individual::get_max_state_size()
        {
            // max stack size is calculated using node arities
            std::map<char, size_t> stack_size;
            std::map<char, size_t> max_stack_size;
            stack_size['f'] = 0;
            stack_size['c'] = 0; 
            stack_size['b'] = 0; 
            max_stack_size['f'] = 0;
            max_stack_size['c'] = 0;
            max_stack_size['b'] = 0;

            for (const auto& n : program)   
            {   	
                ++stack_size[n->otype];

                if ( max_stack_size[n->otype] < stack_size[n->otype])
                    max_stack_size[n->otype] = stack_size[n->otype];

                for (const auto& a : n->arity)
                    stack_size[a.first] -= a.second;       
            }	
            return max_stack_size;
        }

    }

}
