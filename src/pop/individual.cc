/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "individual.h"

namespace FT{   

    namespace Pop{ 
           
        Individual::Individual(){c = 0; dim = 0; eqn=""; parent_id.clear(); parent_id.push_back(-1);}

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
        vector<double> Individual::get_p() const { return p; }     
        
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
            // set weights
            this->w = weights;
        }
        
        double Individual::get_p(const size_t i) const
        {
            /*! @param i index in program 
             * @return weight associated with node */
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
        
        vector<double> Individual::get_p(const vector<size_t>& locs) const
        {
            vector<double> ps;
            for (const auto& el : locs) ps.push_back(get_p(el));
            return ps;
        }
        
        shared_ptr<CLabels> Individual::fit(const Data& d, const Parameters& params, bool& pass)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + get_eqn(), 3);
            Phi = out(d, params);       
            // calculate ML model from Phi
            params.msg("ML training on " + get_eqn(), 3);
            ml = std::make_shared<ML>(params);

            shared_ptr<CLabels> yh = ml->fit(Phi,d.y,params,pass,dtypes);

            this->yhat = ml->labels_to_vector(yh);

            return yh;
        }

        shared_ptr<CLabels> Individual::predict(const Data& d, const Parameters& params)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + get_eqn(), 3);
            // toggle validation
            Phi = out(d, params, true);           // TODO: guarantee this is not changing nodes

            if (Phi.size()==0)
                HANDLE_ERROR_THROW("Phi must be generated before predict() is called\n");
            /* if (drop_idx >= 0)  // if drop_idx specified, mask that phi output */
            /* { */
            /*     cout << "dropping row " + std::to_string(drop_idx) + "\n"; */
            /*     Phi.row(drop_idx) = VectorXd::Zero(Phi.cols()); */
            /* } */
            // calculate ML model from Phi
            params.msg("ML predicting on " + get_eqn(), 3);
            // assumes ML is already trained
            shared_ptr<CLabels> yhat = ml->predict(Phi);
            return yhat;
        }

        VectorXd Individual::predict_drop(const Data& d, const Parameters& params, int drop_idx)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + get_eqn(), 3);
            // toggle validation
            MatrixXd PhiDrop = Phi;           // TODO: guarantee this is not changing nodes
             
            if (Phi.size()==0)
                HANDLE_ERROR_THROW("Phi must be generated before predict_drop() is called\n");
            if (drop_idx >= 0)  // if drop_idx specified, mask that phi output
            {
                if (drop_idx >= PhiDrop.rows())
                    HANDLE_ERROR_THROW("drop_idx ( " + std::to_string(drop_idx) + " > Phi size (" 
                                       + std::to_string(Phi.rows()) + ")\n");
                cout << "dropping row " + std::to_string(drop_idx) + "\n";
                /* PhiDrop.row(drop_idx) = VectorXd::Zero(Phi.cols()); */
                PhiDrop.row(drop_idx).setZero();
            }
            // calculate ML model from Phi
            /* params.msg("ML predicting on " + get_eqn(), 3); */
            // assumes ML is already trained
            VectorXd yh = ml->predict_vector(PhiDrop);
            return yh;
        }

        VectorXd Individual::predict_vector(const Data& d, const Parameters& params)
        {
            return ml->labels_to_vector(this->predict(d,params));
        }
        // calculate program output matrix
        MatrixXd Individual::out(const Data& d, const Parameters& params, bool predict)
        {
            /*!
             * @param d: Data structure
             * @param params: Feat parameters
             * @param predict: if true, this guarantees nodes like split do not get trained
             * @return Phi: n_features x n_samples transformation
             */
             
            State state;
            
            //cout << "In individua.out()\n";
            params.msg("evaluating program " + get_eqn(),3);
            params.msg("program length: " + std::to_string(program.size()),3);
            // evaluate each node in program
            for (const auto& n : program)
            {
                if (n->isNodeTrain()) // learning nodes are set for fit or predict mode
                    dynamic_cast<NodeTrain*>(n.get())->train = !predict;
            	if(state.check(n->arity))
	                n->evaluate(d, state);
                else
                    HANDLE_ERROR_THROW("out() error: node " + n->name + " in " + program_str() + 
                                       " failed arity check\n");
                
            }
            
            // convert state_f to Phi
            params.msg("converting State to Phi",3);
            int cols;
            
            if (state.f.size()==0)
            {
                if (state.b.size() == 0)
                {
                    if (state.c.size() == 0)
                        HANDLE_ERROR_THROW("Error: no outputs in State");
                    
                    cols = state.c.top().size();
                }
                else
                    cols = state.b.top().size();
            }
            else
                cols = state.f.top().size();
                   
            int rows_f = state.f.size();
            int rows_b = state.b.size();
            int rows_c = state.c.size();

            dtypes.clear();        
            Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_c+rows_b, cols);
            
            // add state_f to Phi
            for (unsigned int i=0; i<rows_f; ++i)
            {    
                 ArrayXd Row = ArrayXd::Map(state.f.at(i).data(),cols);
                 clean(Row); // remove nans, set infs to max and min
                 Phi.row(i) = Row;
                 dtypes.push_back('f'); 
            }
            // convert state_b to Phi       
            for (unsigned int i=0; i<rows_b; ++i)
            {
                Phi.row(i+rows_f+rows_c) = ArrayXb::Map(state.b.at(i).data(),cols).cast<double>();
                dtypes.push_back('b');
            }
            // add state_c to Phi
            for (unsigned int i=0; i<rows_c; ++i)
            {    
                 ArrayXd Row = ArrayXi::Map(state.c.at(i).data(),cols).cast<double>();
                 clean(Row); // remove nans, set infs to max and min
                 Phi.row(i+rows_f) = Row;
                 dtypes.push_back('c');
            }       
            return Phi;
        }

        // calculate program output matrix
        MatrixXd Individual::out_trace(const Data& d, const Parameters& params, 
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
            params.msg("evaluating program " + program_str(),3);
            /* params.msg("program length: " + std::to_string(program.size()),3); */

            vector<size_t> roots = program.roots();
            /* cout << "roots: " ; */
            /* for (auto rt : roots) cout << rt << ", "; */
            /* cout << "\n"; */
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
                /* cout << "i = " << i << ", root = " << roots.at(root) << "\n"; */
                if (i > roots.at(root))
                {
                    trace=false;
                    if (root + 1 < roots.size())
                    {
                        ++root; // move to next root
                        // if new root is a Dx node, start storing its subprogram
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
                    {
                        /* cout << "storing trace of " << program.at(i)->name */ 
                        /*      << " for " << program.at(roots.at(root))->name */ 
                        /*      << " with " << program.at(i)->arity['f'] << " arguments\n"; */
                        for (int j = 0; j < program.at(i)->arity['f']; j++) {
                            /* cout << "push back float arg for " << program.at(i)->name << "\n"; */
                            /* cout << "trace_idx: " << trace_idx */ 
                            /*      << ", state_trace size: " << state_trace.size() << "\n"; */
                            state_trace.at(trace_idx).f.push_back(state.f.at(state.f.size() - 
                                                             (program.at(i)->arity['f'] - j)));
                        }
                        
                        for (int j = 0; j < program.at(i)->arity['b']; j++) {
                            /* cout << "push back bool arg for " << program.at(i)->name << "\n"; */
                            state_trace.at(trace_idx).b.push_back(state.b.at(state.b.size() - 
                                                             (program.at(i)->arity['b'] - j)));
                        }

                        for (int j = 0; j < program.at(i)->arity['c']; j++) {
                            /* cout << "push back categorial arg for " << program.at(i)->name << "\n"; */
                            state_trace.at(trace_idx).c.push_back(state.c.at(state.c.size() - 
                                                             (program.at(i)->arity['c'] - j)));
                        }
                    }
	                program.at(i)->evaluate(d, state);
                    program.at(i)->visits = 0;
	            }
                else
                    HANDLE_ERROR_THROW("out() error: node " + program.at(i)->name + " in " + program_str() + " is invalid\n");
            }
            
            // convert state_f to Phi
            params.msg("converting State to Phi",3);
            int cols;
            if (state.f.size()==0)
            {
                if (state.c.size() == 0)
                {
                    if (state.b.size() == 0)
                        HANDLE_ERROR_THROW("Error: no outputs in State");
                    
                    cols = state.b.top().size();
                }
                else
                    cols = state.c.top().size();
            }
            else
                cols = state.f.top().size();
                   
            int rows_f = state.f.size();
            int rows_b = state.b.size();
            int rows_c = state.c.size();
            
            dtypes.clear();        
            Matrix<double,Dynamic,Dynamic,RowMajor> Phi (rows_f+rows_c+rows_b, cols);
            
            // add state_f to Phi
            for (unsigned int i=0; i<rows_f; ++i)
            {    
                 ArrayXd Row = ArrayXd::Map(state.f.at(i).data(),cols);
                 clean(Row); // remove nans, set infs to max and min
                 Phi.row(i) = Row;
                 dtypes.push_back('f'); 
            }
            
            // convert state_b to Phi       
            for (unsigned int i=0; i<rows_b; ++i)
            {
                Phi.row(i+rows_f+rows_c) = ArrayXb::Map(state.b.at(i).data(),cols).cast<double>();
                dtypes.push_back('b');
            }

            // add state_c to Phi
            for (unsigned int i=0; i<rows_c; ++i)
            {    
                 ArrayXd Row = ArrayXi::Map(state.c.at(i).data(),cols).cast<double>();
                 clean(Row); // remove nans, set infs to max and min
                 Phi.row(i+rows_f) = Row;
                 dtypes.push_back('c'); 
            }       
            //Phi.transposeInPlace();
            return Phi;
        }
        
        // return symbolic representation of program 
        string Individual::get_eqn()
        {
            //cout << "Called get_eqn()"<<"\n";
            if (eqn.empty())               // calculate eqn if it doesn't exist yet 
            {
                //cout << "eqn is empty\n";
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
                    eqn += "[" + s + "]";
                for (auto s : state.bs) 
                    eqn += "[" + s + "]";
                for (auto s : state.cs)
                    eqn += "[" + s + "]";              
                for (auto s : state.zs) 
                    eqn += "[" + s + "]";
            }
            
            //cout << "returning equation as "<<eqn << "\n"; 
            return eqn;
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
    }

}
