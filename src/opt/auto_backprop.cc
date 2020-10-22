
#include "auto_backprop.h"

/**
TODO
------------------------
Integrate vectorList
Integrate pointers?
TODO Make it so stops traversing once it hits a non-differentiable node and then goes upstream and finds another branch to traverse
**/

namespace FT {
    
    namespace Opt{
	    
        AutoBackProp::AutoBackProp(string scorer, int iters, float n, float a) 
        {
		    /* this->program = program.get_data(); */
            score_hash["mse"] = &Eval::squared_difference;
            score_hash["log"] =  &Eval::log_loss; 
            score_hash["multi_log"] =  &Eval::multi_log_loss;
            score_hash["fpr"] =  &Eval::log_loss;
            score_hash["zero_one"] = &Eval::log_loss;

            d_score_hash["mse"] = &Eval::d_squared_difference;
            d_score_hash["log"] =  &Eval::d_log_loss; 
            d_score_hash["multi_log"] =  &Eval::d_multi_log_loss;
            d_score_hash["fpr"] =  &Eval::d_log_loss;
            d_score_hash["zero_one"] = &Eval::d_log_loss;            

            this->d_cost_func = d_score_hash.at(scorer); 
            this->cost_func = score_hash.at(scorer); 
            
            /* this->X = X; */
		    /* this->labels = labels; */
		    this->iters = iters;
		    this->n = n;
            this->epT = 0.01*this->n;   // min learning rate
		    this->a = a;
	    }

	    void AutoBackProp::print_weights(NodeVector& program) {
	        for (const auto& p : program) 
            {
		        cout << "( " << p->name;
		        if (p->isNodeDx()) {
                    
			        NodeDx* dNode = dynamic_cast<NodeDx*>(p.get());
			        for (int i = 0; i < dNode->arity.at('f'); i++) {
				        cout << "," << dNode->W.at(i);
			        }
                    dNode = nullptr;
		        }

		        cout << " ) ";
	        }
            /* cout << "\n"; */
        }

        void AutoBackProp::run(Individual& ind, const Data& d,
                                const Parameters& params)
        {
            vector<size_t> roots = ind.program.roots();
            float min_loss;
            float current_loss, current_val_loss;
            vector<vector<float>> best_weights;
            // split up the data so we have a validation set
            DataRef BP_data(d.X, d.y, d.Z, d.classification);
            BP_data.train_test_split(true, 0.5);
            // set up batch data
            MatrixXf Xb, Xb_v;
            VectorXf yb, yb_v;
            /* std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Zb, Zb_v; */
            LongData Zb, Zb_v;
            /* cout << "y: " << d.y.transpose() << "\n"; */ 
            Data batch_data(Xb, yb, Zb, params.classification);
            /* Data db_val(Xb_v, yb_v, Zb_v, params.classification); */
            /* db_val.set_validation();    // make this a validation set */
            // if batch size is 0, set batch to 20% of training data
            int batch_size = params.bp.batch_size > 0? 
                params.bp.batch_size : .2*BP_data.t->y.size(); 
            /* d.get_batch(db_val, );     // draw a batch for the validation data */
            // number of iterations to allow validation fitness to not improve
            int patience = 3;               
            int missteps = 0;

            this->epk = n;  // starting learning rate
            /* logger.log("running backprop on " + ind.get_eqn(), 2); */
            /* cout << ind.get_eqn() << endl; */
            logger.log("=========================",4);
            logger.log("Iteration,Train Loss,Val Loss,Weights",4);
            logger.log("=========================",4);
            for (int x = 0; x < this->iters; x++)
            {
                logger.log("get batch",3);
                // get batch data for training
                BP_data.t->get_batch(batch_data, batch_size); 
                /* cout << "batch_data.y: " */ 
                /*      << batch_data.y.transpose() << "\n"; */ 
                // Evaluate forward pass
                MatrixXf Phi; 
                logger.log("forward pass",3);
                vector<Trace> stack_trace = forward_prop(ind, batch_data, 
                        Phi, params);
                // Evaluate ML model on Phi
                bool pass = true;
                auto ml = std::make_shared<ML>(params.ml, true, 
                        params.classification, params.n_classes);

                logger.log("ml fit",3);
                shared_ptr<CLabels> yhat = ml->fit(Phi,
                        batch_data.y,params,pass,ind.dtypes);
                
                
                if (!pass || stack_trace.size() ==0 )
                    break;

                vector<float> Beta = ml->get_weights();
                /* cout << "cost func\n"; */
                current_loss = this->cost_func(batch_data.y, yhat, 
                        params.class_weights).mean();

                // Evaluate backward pass
                size_t s = 0;
                for (int i = 0; i < stack_trace.size(); ++i)
                {
                    while (!ind.program.at(roots.at(s))->isNodeDx()) ++s;
                    /* cout << "roots.at(s): " << roots.at(s) << endl; */
                    /* cout << "running backprop on " */ 
                    /*      << ind.get_eqn() << "\n"; */ 
                    /* << " from " */
                    /*      << roots.at(s) << " to " */  
                    /*      << ind.program.subtree(roots.at(s)) << "\n"; */
                    
                    backprop(stack_trace.at(i), ind.program, 
                            ind.program.subtree(roots.at(s)), 
                            roots.at(s), Beta.at(s)/ml->N.scale.at(s), 
                            yhat, batch_data, params.class_weights);
                }

                // check validation fitness for early stopping
                MatrixXf Phival = ind.out((*BP_data.v));
                logger.log("checking validation fitness",3);
                /* cout << "Phival: " << Phival.rows() 
                 * << " x " << Phival.cols() << "\n"; */
                /* cout << "y_val\n"; */
                shared_ptr<CLabels> y_val = ml->predict(Phival);
                current_val_loss = this->cost_func(BP_data.v->y, y_val, 
                        params.class_weights).mean();
                if (x==0 || current_val_loss < min_loss)
                {
                    min_loss = current_val_loss;
                    best_weights = ind.program.get_weights();
                }
                else
                {
                    ++missteps;
                    /* cout << "missteps: " << missteps << "\n"; */
                    logger.log("",3);           // update learning rate
                }
                // early stopping trigger
                if (missteps == patience 
                        || std::isnan(min_loss) 
                        || std::isinf(min_loss)
                        || min_loss <= NEAR_ZERO)       
                    break;
                else
                    logger.log("min loss: " + std::to_string(min_loss), 3);

                float alpha = float(x)/float(iters);

                this->epk = (1 - alpha)*this->epk + alpha*this->epT;  
                /* this->epk = this->epk + this->epT; */ 
                /* cout << "epk: " << this->epk << "\n"; */
                if (params.verbosity>3)
                {
                    cout << x << "," 
                     << current_loss << "," 
                     << current_val_loss << ",";
                     print_weights(ind.program);
                }
            }
            logger.log("",4);
            logger.log("=========================",4);
            logger.log("done=====================",4);
            logger.log("=========================",4);
            ind.program.set_weights(best_weights);
        }
        
        // forward pass
        vector<Trace> AutoBackProp::forward_prop(Individual& ind, const Data& d,
                                                 MatrixXf& Phi, const Parameters& params) 
        {
            /* cout << "Forward pass\n"; */
            // Iterate through all the nodes evaluating and tracking ouputs
            vector<Trace> stack_trace;
            Phi = ind.out_trace(d, stack_trace);
            // Use stack_f and execution stack to avoid issue of branches affecting what elements 
            // appear before a node 
            /* cout << "Returning forward pass.\n"; */
            return stack_trace;
        }   
        // Updates stacks to have proper value on top
        void AutoBackProp::next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                                       vector<ArrayXf>& derivatives) 
        {
            // While there are still nodes with branches to explore
            if(!executing.empty()) {
                // Declare variable to hold node and its associated derivatives
                BP_NODE bp_node = pop<BP_NODE>(&executing); // Check first element
                // Loop until branch to explore is found
                while (bp_node.deriv_list.empty() && !executing.empty()) {
                    bp_node = pop<BP_NODE>(&executing); // Get node and its derivatves

                    // For some reason this function is not removing element from the stack
                    pop<ArrayXf>(&derivatives); // Remove associated gradients from stack
                    if (executing.empty()) {
                        return;
                    }
                }
                
                // Should now have the next parent node and derivatves (stored in bp_node)
                if (!bp_node.deriv_list.empty()) 
                {
                    bp_program.push_back(bp_node.n);
                    // Pull derivative from front of list due to how we stored them earlier
                    derivatives.push_back(pop_front<ArrayXf>(&(bp_node.deriv_list)));                 
                    // Push it back on the stack in order to sync all the stacks
                    executing.push_back(bp_node);             
                }
            }
        }

        // Compute gradients and update weights 
        void AutoBackProp::backprop(Trace& stack, NodeVector& program, int start, int end, 
                                    float Beta, shared_ptr<CLabels>& yhat, 
                                    const Data& d,
                                    vector<float> sw)    
        {
            /* cout << "Backward pass \n"; */
            vector<ArrayXf> derivatives;
            // start with derivative of cost function wrt ML output times dyhat/dprogram output, which
            // is equal to the weight the model assigned to this subprogram (Beta)
            // push back derivative of cost function wrt ML output
            /* cout << "Beta: " << Beta << "\n"; */ 
            derivatives.push_back(this->d_cost_func(d.y, yhat, sw).array() * Beta); //*phi.array()); 
            /* cout << "Cost derivative: " << derivatives[derivatives.size() -1 ]<< "\n"; */ 
            // Working according to test program */
            /* pop<ArrayXf>(&f_stack); // Get rid of input to cost function */
            vector<BP_NODE> executing; // Stores node and its associated derivatves
            // Currently I don't think updates will be saved, might want a pointer of nodes so don't 
            // have to restock the list
            // Program we loop through and edit during algorithm (is this a shallow or deep copy?)
            /* cout << "copy program \n"; */
            vector<Node*> bp_program = program.get_data(start, end);         
            /* cout << "Initializing backprop systems.\n"; */
            while (bp_program.size() > 0) {
                /* cout << "Size of program: " << bp_program.size() << "\n"; */
                Node* node = pop<Node*>(&bp_program);
                /* cout << "Evaluating: " << node->name << "\n"; */
                /* cout << "executing stack: " ; */ 
                /* for (const auto& bpe : executing) cout << bpe.n->name << " " ; cout << "\n"; */
                /* cout << "bp_program: " ; */ 
                /* for (const auto& bpe : bp_program) cout << bpe->name << " " ; cout << "\n"; */
                /* cout << "derivatives size: " << derivatives.size() << "\n"; */ 
                vector<ArrayXf> n_derivatives;

                if (node->isNodeDx() && node->visits == 0 && node->arity.at('f') > 0) {
                    NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
                    /* cout << "evaluating derivative\n"; */
                    // Calculate all the derivatives and store them, then update all the weights and throw away the node
                    for (int i = 0; i < node->arity.at('f'); i++) {
                        dNode->derivative(n_derivatives, stack, i);
                    }
                    /* cout << "updating derivatives\n"; */
                    dNode->update(derivatives, stack, this->epk, this->a);
                    // dNode->print_weight();
                    /* cout << "popping input arguments\n"; */
                    // Get rid of the input arguments for the node
                    for (int i = 0; i < dNode->arity.at('f'); i++) {
                        pop<ArrayXf>(&stack.f);
                    }
                    for (int i = 0; i < dNode->arity.at('b'); i++) {
                        pop<ArrayXb>(&stack.b);
                    }
                    for (int i = 0; i < dNode->arity.at('c'); i++) {
                        pop<ArrayXi>(&stack.c);
                    }
                    if (!n_derivatives.empty()) {
                        derivatives.push_back(pop_front<ArrayXf>(&n_derivatives));
                    }

                    executing.push_back({dNode, n_derivatives});
                }
                /* else */
                /*     cout << "not NodeDx or visits reached or no floating arity\n"; */
                /* cout << "next branch\n"; */
                // Choosing how to move through tree
                if (node->arity.at('f') == 0 || !node->isNodeDx()) {
            
                    // Clean up gradients and find the parent node
                    /* cout << "popping derivatives\n"; */
                    if (!derivatives.empty())
                        pop<ArrayXf>(&derivatives);	// TODO check if this fixed
                    next_branch(executing, bp_program, derivatives);
                } 
                else 
                {
                    node->visits += 1;
                    if (node->visits > node->arity.at('f')) 
                    {
                        next_branch(executing, bp_program, derivatives);
                    }
                }
            }

            // point bp_program to null
            for (unsigned i = 0; i < bp_program.size(); ++i)
                bp_program.at(i) = nullptr;

            /* cout << "Backprop terminated\n"; */
            //print_weights(program);
        }
    }    
}

