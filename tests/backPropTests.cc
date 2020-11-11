#include "testsHeader.h"

using namespace Opt;

/**
Notes
Add import from util for 2d Gauss
**/

bool isNodeDx(Node* n) {
	return NULL != dynamic_cast<NodeDx*>(n); 
}

ArrayXf limited(ArrayXf x)
{
    //cout << "\n From limited function\n INPUT \n"<<x<<"\n";
    x = (isnan(x)).select(0,x);
    x = (x < MIN_FLT).select(MIN_FLT,x);
    x = (x > MAX_FLT).select(MAX_FLT,x);
    
    //cout << "\n From limited function\n OUTPUT \n"<<x<<"\n";
    return x;
};

/*ArrayXf evaluateProgram(NodeVector& program, MatrixXf data, VectorXf labels)

	Stacks stack;

	std::cout << "Running evaluation.\n";
	// Iterate through program and calculate results 
	for (const auto& n : program) {
		std::cout << "Running: " << n->name << "\n";
		n->evaluate(data, labels, z, stack);
		std::cout << "result:" << stack_f[stack_f.size() - 1] << "\n--";
	}

	std::cout << "Returning result.\n" << stack_f[stack_f.size() - 1] << "\n-----------------\n";
	return pop(stack_f);
}*/

Node* parseToNode(std::string token) {
	if (token == "+") {
    	return new NodeAdd({1.0, 1.0});
    } else if (token == "-") {
    	return new NodeSubtract({1.0, 1.0});
    } else if (token == "/") {
    	return new NodeDivide({1.0, 1.0});
    } else if (token == "*") {
    	return new NodeMultiply({1.0, 1.0});
    } else if (token == "cos") {
    	return new NodeCos({1.0});
    } else if (token == "sin") {
    	return new NodeSin({1.0});
   	} else if (token == "tanh") {
    	return new NodeTanh({1.0});
    } else if (token == "x0") {
    	return new NodeVariable<float>(0);
    } else if (token == "x1") {
    	return new NodeVariable<float>(1);
    } else if (token == "c") {
    return new NodeConstant(1.0);
    } else if (token == "exponent") {
    	return new NodeExponent({1.0, 1.0});
    } else if (token == "exp") {
    	return new NodeExponential({1.0});
    } else if (token == "log") {
    	return new NodeLog({1.0});
    } else if (token == "sqrt") {
    	return new NodeSqrt({1.0});
    } else if (token == "relu") {
    	return new NodeRelu({1.0});
    } else if (token == "sign") {
    	return new NodeSign();
    } else if (token == "logit") {
    	return new NodeLogit({1.0});
    } else if (token == "gauss") {
        return new NodeGaussian({1.0});
    } else if (token == "max") {
    	return new NodeMax();
    } else if (token == "xor") {
    	return new NodeXor();
    } else if (token == "step") {
    	return new NodeStep();
    }
}

class TestBackProp
{
    public:
    
        TestBackProp(int iters=1000, float n=0.1, float a=0.9)
        {
            this->iters = iters;
		    this->n = n;
            this->epT = 0.01*this->n;   // min learning rate
		    this->a = a;
		    this->engine = new AutoBackProp("mse", iters, n, a);
        }
        
        void run(Individual& ind, const Data& d,
                            const Parameters& params)
        {
            vector<size_t> roots = ind.program.roots();
            float min_loss;
            float current_loss, current_val_loss;
            vector<vector<float>> best_weights;
            // batch data
            MatrixXf Xb, Xb_v;
            VectorXf yb, yb_v;
            std::map<string, std::pair<vector<ArrayXf>, 
                vector<ArrayXf> > > Zb, Zb_v;
            /* cout << "y: " << d.y.transpose() << "\n"; */ 
            Data db(Xb, yb, Zb, params.classification);
            Data db_val(Xb_v, yb_v, Zb_v, params.classification);
            db_val.set_validation();    // make this a validation set
            // draw a batch for the validation data
            d.get_batch(db_val, params.bp.batch_size);     

            // number of iterations to allow validation fitness to not improve
            int patience = 3;   
            int missteps = 0;

            this->epk = n;  // starting learning rate
            /* cout << "running backprop on " << ind.get_eqn() << endl; */
//            logger.log("=========================",3);
//            logger.log("Iteration,Train Loss,Val Loss,Weights",3);
//            logger.log("=========================",3);
            for (int x = 0; x < this->iters; x++)
            {
                /* cout << "\n\nIteration " << x << "\n"; */
                /* cout << "get batch\n"; */
                // get batch data for training
                d.get_batch(db, params.bp.batch_size); 
                /* cout << "db.y: " << db.y.transpose() << "\n"; */ 
                // Evaluate forward pass
                MatrixXf Phi; 
                /* cout << "forward pass\n"; */
                vector<Trace> stack_trace = engine->forward_prop(ind, db, 
                        Phi, params);

                // Evaluate backward pass
                size_t s = 0;
                for (int i = 0; i < stack_trace.size(); ++i)
                {
                    while (!ind.program.at(roots[s])->isNodeDx()) ++s;
                    /* cout << "running backprop on " */ 
                    /*     << ind.program_str() << " from " */
                    /*     << roots.at(s) << " to " */ 
                    /*     << ind.program.subtree(roots.at(s)) << "\n"; */
                    
                    backprop(stack_trace.at(i), ind.program, 
                            ind.program.subtree(roots.at(s)), 
                            roots.at(s), 1.0, Phi.row(0), db, 
                            params.class_weights);
                }

                /* cout << "squared_difference\n"; */

                MatrixXf Phival = ind.out(db_val);

                current_val_loss = squared_difference(db_val.y, 
                        Phival.row(0)).mean();
                
                if (x==0 || current_val_loss < min_loss)
                {
//                    logger.log("current value loss: " 
//                    + std::to_string(current_val_loss), 3);
                    min_loss = current_val_loss;
                    best_weights = ind.program.get_weights();
//                    logger.log("new min loss: " 
//                    + std::to_string(min_loss), 3);
                }
                else
                {
                    ++missteps;
//                    cout << "missteps: " << missteps << "\n";
//                    logger.log("current value loss: " 
//                    + std::to_string(current_val_loss), 3);
//                    logger.log("new min loss: " 
//                    + std::to_string(min_loss), 3);
//                    logger.log("",3);           // update learning rate
                }
                
                /* float alpha = float(x)/float(iters); */
                /* this->epk = (1 - alpha)*this->epk + alpha*this->epT; */  
//                cout << "Verbosity is " << params.verbosity << "\n";
//                if (params.verbosity>2)
//                {
                    logger.log( to_string(x) + ", " 
                     + to_string(current_loss) + ", " 
                     + to_string(current_val_loss) ,
                     3);
//                }
            }
            logger.log("",3);
            logger.log("=========================",3);
            logger.log("done=====================",3);
            logger.log("=========================",3);
            ind.program.set_weights(best_weights);
        }
        
        void backprop(Trace& stack, NodeVector& program, int start, int end, 
                                float Beta, const VectorXf& yhat, 
                                const Data& d,
                                vector<float> sw)    
        {
            /* cout << "Backward pass \n"; */
            vector<ArrayXf> derivatives;
            // start with derivative of cost function wrt ML output times dyhat/dprogram output, which
            // is equal to the weight the model assigned to this subprogram (Beta)
            // push back derivative of cost function wrt ML output
            /* cout << "Beta: " << Beta << "\n"; */ 
            derivatives.push_back(d_squared_difference(d.y, yhat).array() * Beta); //*phi.array()); 
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

                if (node->isNodeDx() && node->visits == 0 && node->arity['f'] > 0) {
                    NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
                    /* cout << "evaluating derivative\n"; */
                    // Calculate all the derivatives and store them, then update all the weights and throw away the node
                    for (int i = 0; i < node->arity['f']; i++) {
                        dNode->derivative(n_derivatives, stack, i);
                    }
                    /* cout << "updating derivatives\n"; */
                    dNode->update(derivatives, stack, this->epk, this->a);
                    // dNode->print_weight();
                    /* cout << "popping input arguments\n"; */
                    // Get rid of the input arguments for the node
                    for (int i = 0; i < dNode->arity['f']; i++) {
                        pop<ArrayXf>(&stack.f);
                    }
                    for (int i = 0; i < dNode->arity['c']; i++) {
                        pop<ArrayXi>(&stack.c);
                    }
                    for (int i = 0; i < dNode->arity['b']; i++) {
                        pop<ArrayXb>(&stack.b);
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
                if (node->arity['f'] == 0 || !node->isNodeDx()) {
            
                    // Clean up gradients and find the parent node
                    /* cout << "popping derivatives\n"; */
                    if (!derivatives.empty())
                        pop<ArrayXf>(&derivatives);	// TODO check if this fixed
                    engine->next_branch(executing, bp_program, derivatives);
                } 
                else 
                {
                    node->visits += 1;
                    if (node->visits > node->arity['f']) 
                    {
                        engine->next_branch(executing, bp_program, derivatives);
                    }
                }
            }

            // point bp_program to null
            for (unsigned i = 0; i < bp_program.size(); ++i)
                bp_program[i] = nullptr;

            /* cout << "Backprop terminated\n"; */
            //print_weights(program);
        }
        
    private:
        int iters;
        float n;
        float epT;
        float a;
        float epk;
        AutoBackProp* engine;
        
};

NodeVector programGen(std::string txt) {
	NodeVector program;

	char ch = ' ';
	size_t pos = txt.find( ch );
    size_t initialPos = 0;

    // Decompose statement
    std::string token;
    while( pos != std::string::npos ) {
    	token = txt.substr( initialPos, pos - initialPos );

        program.push_back(unique_ptr<Node>(parseToNode(token)));

        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    token = txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 );
    program.push_back(unique_ptr<Node>(parseToNode(token)));

    return program;
}

Individual testDummyProgram(NodeVector p0, Data data, int iters, 
        VectorXf& yhat) 
{
	
	std::cout << "Testing program: [";
	
	for (const auto& n : p0) {
		std::cout << n->name << ", ";
	}
	std::cout << "]\n";
	
	std::cout << "Number of iterations are "<< iters <<"\n";

	// Params
	float learning_rate = 0.25;
    int bs = 100; 
    Individual ind;
    ind.program = p0;
    Feat feat;
    feat.set_batch_size(bs);
    /* feat.set_verbosity(3); */
    
    feat.set_shuffle(false);
                          
    /* std::cout << "TestBackProp\n"; */
	TestBackProp* engine = new TestBackProp(iters, learning_rate, 0.9);	

    /* std::cout << "engine->run\n"; */
    engine->run(ind, data, feat.params); // Update pointer to NodeVector internally
    /* std::cout << "ind.out()"; */
    MatrixXf Phi = ind.out(data); 
    yhat = Phi.row(0); 
	return ind;

	// Make sure internal NodeVector updated
}

TEST(BackProp, SumGradient)
{   
    // Create input data and labels
	MatrixXf X(2, 10);
	VectorXf y(10);
	
	X.row(0) << -0.44485052, -0.49109715,  0.88231917,  0.94669031, -0.80300709,
       -0.581858  , -0.91693663, -0.98437617, -0.52860637, -0.89671113;
    X.row(1) << 0.89560483,  0.87110481, -0.47065155,  0.32214509,  0.59596947,
        0.81329039,  0.39903285,  0.17607827,  0.84886707, -0.44261626;

    /* y << 1.79711347,  1.63112011,  0.35268371,  2.85981589,  0.18189424, */
    /*     1.27615517, -0.63677472, -1.44051753,  1.48938848, -3.12127104; */
        
    y = 2*X.row(0).array()+3*X.row(1).array();
    /* cout << "X[0]: " << X.row(0) << "\n"; */
    /* cout << "X[1]: " << X.row(1) << "\n"; */
    /* cout << "y: " << y.transpose() << "\n"; */
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x1 x0 +");
    VectorXf yhat; 
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
    vector<float> What(2);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
			}
		}
		std::cout << "\n";
	}
    cout << y - yhat << "\n";
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001);
    ASSERT_LE(fabs(What[0]-2),0.000001);
    ASSERT_LE(fabs(What[1]-3),0.000001);
}

TEST(BackProp, SubtractGradient)
{   
    // Create input data and labels
	MatrixXf X(2, 10);
	VectorXf y(10);
	
	X.row(0) << -0.44485052, -0.49109715,  0.88231917,  0.94669031, -0.80300709,
       -0.581858  , -0.91693663, -0.98437617, -0.52860637, -0.89671113;
    
    X.row(1) << 0.89560483,  0.87110481, -0.47065155,  0.32214509,  0.59596947,
        0.81329039,  0.39903285,  0.17607827,  0.84886707, -0.44261626;

    y = 2*X.row(0)-3*X.row(1);
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x1 x0 -");
    VectorXf yhat; 
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	vector<float> What(2);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
				/* std::cout << "What[i]: " << What[i] << "\n"; */
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001);
    ASSERT_LE(fabs(What[0]-2),0.000001);
    ASSERT_LE(fabs(What[1]-3),0.000001);

}

TEST(BackProp, MultiplyGradient)
{   
    // Create input data and labels
	MatrixXf X(2, 10);
	VectorXf y(10);
	
	X.row(0) << -0.44485052, -0.49109715,  0.88231917,  0.94669031, -0.80300709,
       -0.581858  , -0.91693663, -0.98437617, -0.52860637, -0.89671113;
    X.row(1) << 0.89560483,  0.87110481, -0.47065155,  0.32214509,  0.59596947,
        0.81329039,  0.39903285,  0.17607827,  0.84886707, -0.44261626;

    y = (2*X.row(0))*(3*X.row(1));
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 x1 *");
    VectorXf yhat; 
    Individual ind = testDummyProgram(program, data, 1000, yhat);

    
    std::cout << "test program returned:\n";
	vector<float> What(2);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    /* ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001); */
    // What[0] and What[1] should be equivalent since they are completely correlated here
    ASSERT_LE(fabs(What[0]-What[1]),0.000001);
}

TEST(BackProp, DivideGradient)
{   
    // Create input data and labels
	MatrixXf X(2, 10);
	VectorXf y(10);
	
	X.row(0) << -0.44485052, -0.49109715,  0.88231917,  0.94669031, -0.80300709,
       -0.581858  , -0.91693663, -0.98437617, -0.52860637, -0.89671113;
    X.row(1) << 0.89560483,  0.87110481, -0.47065155,  0.32214509,  0.59596947,
        0.81329039,  0.39903285,  0.17607827,  0.84886707, -0.44261626;
 
    float Wnom = 1.234;
    float Wden = 0.9876;
    /* y = ((Wnom*X.row(0).array())/(Wden*X.row(1).array())); */
    // y = 1/ (x)
    /* y = Wnom/(Wden*X.row(0).array()); */
    y = Wnom/Wden*ArrayXf::Ones(X.rows());
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x1 x1 /");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 10000, yhat);
    
    std::cout << "test program returned:\n";
    vector<float> What(2);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    /* ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001); */
    cout << "what[0]/what[1]: " << What[0]/What[1] << "\n";
    cout << "Wnom/Wden : " << Wnom/Wden << "\n";
    // The ratio of the true weights should match the ratio of the found weights
    ASSERT_LE(fabs(Wnom/Wden - What[0]/What[1]),0.001);
    /* ASSERT_LE(What[1]-Wden,0.001); */
}

TEST(BackProp, SinGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = sin(2x)
    y << 0.91615131, 0.99999293, 0.99480094, 0.98852488, 0.51932576,
       0.92795746, 0.29467693, 0.93974488, 0.55849396, 0.67245834;
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 sin");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001);
}

TEST(BackProp, CosGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = cos(2x)
    y << -0.40083261, -0.00376088,  0.10183854,  0.15105817,  0.85457636,
        0.37268613,  0.95559694, -0.34187653,  0.82950858,  0.74013498;
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 cos");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001);
}

TEST(BackProp, TanhGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = tanh(2x)
    y << 0.96282281, 0.91774763, 0.89934465, 0.88942308, 0.49756278,
       0.83023564, 0.29050472, 0.95789335, 0.53174082, 0.62764766;
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 tanh");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
}

TEST(BackProp, ExpGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = exp(2x)
    y << exp(1.2345*X.row(0).array());
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 exp");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
}

TEST(BackProp, LogGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = log(2x)
    y << log(2*X.row(0).array());
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 log");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
}

TEST(BackProp, SqrtGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0.78727861, 0.73439047, 0.70957884, 0.27303089,
       0.59444715, 0.14955871, 0.95985467, 0.29628456, 0.36876264;
    
    //y = sqrt(2x)
    y << sqrt(2*X.row(0).array());
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 sqrt");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    ASSERT_LE((y-yhat).array().pow(2).sum(),0.000001);
}

TEST(BackProp, ReluGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.9916109 , 0, 0.73439047, 0.70957884, 0,
       0.59444715, 0, 0.95985467, 0, 0;
    
    //y = relu(2x)
    y << 1.9832218, 0.01, 1.46878094, 1.41915768, 0.01,
         1.1888943, 0.01, 1.91970934, 0.01, 0.01;
        
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 relu");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */
    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
}

TEST(BackProp, LogitGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.01933084, 0.46202196, 0.26687028, 0.31371363, 0.27296074,
       0.37672478, 0.05773657, 0.36211793, 0.0161587 , 0.02614942;
    
    //y = logit(2x)
    /* y << -3.21347748,  2.49860441,  0.13516769,  0.52119546,  0.18420504, */
    /*     1.11709547, -2.03601496,  0.9655712 , -3.39929844, -2.89706515; */
    float Wtarget = 1.234;
    y = 1/(1 + exp(-Wtarget*X.row(0).array()));

    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 logit");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 2000, yhat);
    
    std::cout << "test program returned:\n";
    vector<float> What(1);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */

    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
    ASSERT_LE(fabs(What[0]-Wtarget),0.01);
}

TEST(BackProp, GaussGradient)
{
    // Create input data and labels
	MatrixXf X(1, 10);
	VectorXf y(10);
	X.row(0) << 0.01933084, 0.46202196, 0.26687028, 0.31371363, 0.27296074,
       0.37672478, 0.05773657, 0.36211793, 0.0161587 , 0.02614942;
    
    float Wtarget = 2;
    y = exp(-pow(Wtarget - X.row(0).array(), 2));

    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z; 
    
    Data data(X, y, Z);
    
    NodeVector program = programGen("x0 gauss");
    
    VectorXf yhat;
    Individual ind = testDummyProgram(program, data, 1000, yhat);
    
    std::cout << "test program returned:\n";
    vector<float> What(1);
	for (const auto& n : ind.program) {
		std::cout << n->name << ": ";
		NodeDx* nd = dynamic_cast<NodeDx*>(n.get());
		if (nd != NULL) {
			std::cout << " with weight";
			for (int i = 0; i < nd->arity['f']; i++) {
				std::cout << " " << nd->W[i];
                What[i] = nd->W[i];
			}
		}
		std::cout << "\n";
	}

    /* cout << y - yhat << "\n"; */

    ASSERT_LE((y-yhat).array().pow(2).sum(),0.00001);
    ASSERT_LE(fabs(What[0]-Wtarget),0.01);
}

TEST(BackProp, DerivativeTest)
{
	Trace trace;
	//vector<ArrayXf> inputs;
	ArrayXf input1(5,1);
	input1(0,0) = 0;
	input1(1,0) = 1;
	input1(2,0) = 2;
	input1(3,0) = 3;
	input1(4,0) = 4;
	ArrayXf input2(5,1);
	input2(0,0) = 4;
	input2(1,0) = 3;
	input2(2,0) = 2;
	input2(3,0) = 1;
	input2(4,0) = 0;
	trace.f.push_back(input2);
	trace.f.push_back(input1);

	// ADD NODE CHECK -------------------------------------------------------------------------------
	NodeDx* toTest = new NodeAdd();
	
	// Derivative wrt to first input
	ArrayXf expectedDerivative(5, 1);
	expectedDerivative(0,0) = toTest->W[0];
	expectedDerivative(1,0) = toTest->W[0];
	expectedDerivative(2,0) = toTest->W[0];
	expectedDerivative(3,0) = toTest->W[0];
	expectedDerivative(4,0) = toTest->W[0];
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 
              0.0001);
		
	expectedDerivative(0,0) = toTest->W[1];
	expectedDerivative(1,0) = toTest->W[1];
	expectedDerivative(2,0) = toTest->W[1];
	expectedDerivative(3,0) = toTest->W[1];
	expectedDerivative(4,0) = toTest->W[1];

	// Derivative wrt to second input
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 
              0.0001);

	// Derivative wrt to first weight
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);	
    
	// Derivative wrt to second weight
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
    

	// SUB NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeSubtract({1,1});
    
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -1;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -1;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
		
    expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;

    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
    
    expectedDerivative(0,0) = -4;
	expectedDerivative(1,0) = -3;
	expectedDerivative(2,0) = -2;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);

	// MULT NODE CHECK-------------------------------------------------------------------------------
	toTest = new NodeMultiply({1,1});
	
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 4;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 0;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);
    
    // DIV NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeDivide({1,1});
	
	expectedDerivative(0,0) = 1.0/4;	// Div by 0 (limited to 0)
	expectedDerivative(1,0) = 1.0/3;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1.0/1;
	expectedDerivative(4,0) = MAX_FLT;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	 
	expectedDerivative(0,0) = -0.0/16;	// Div by 0
	expectedDerivative(1,0) = -1.0/9;
	expectedDerivative(2,0) = -2.0/4;
	expectedDerivative(3,0) = -3.0/1;
	expectedDerivative(4,0) = MIN_FLT; 
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 0.0/4;	// Div by 0
	expectedDerivative(1,0) = 1.0/3;
	expectedDerivative(2,0) = 2.0/2;
	expectedDerivative(3,0) = 3.0/1;
	expectedDerivative(4,0) = MAX_FLT;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -0.0/4;	//Div by 0
	expectedDerivative(1,0) = -1.0/3;
	expectedDerivative(2,0) = -2.0/2;
	expectedDerivative(3,0) = -3.0/1;
	expectedDerivative(4,0) = -MAX_FLT;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);

	// x^y NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeExponent({1.0,1.0});
	
	expectedDerivative(0,0) = 4 * pow(0,4)/0; //div by 0 
	expectedDerivative(1,0) = 3 * pow(1,3)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/2;
	expectedDerivative(3,0) = 1 * pow(3,1)/3;
	expectedDerivative(4,0) = 0 * pow(4,0)/4;
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 1 * pow(0,4) * log(0); //log 0 
    expectedDerivative(1,0) = 1 * pow(1,3) * log(1);
	expectedDerivative(2,0) = 1 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(4,0) = 1 * pow(4,0) * log(4);
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 0 * pow(4,0)/1;
	expectedDerivative(1,0) = 1 * pow(3,1)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/1;
	expectedDerivative(3,0) = 3 * pow(1,3)/1;
	expectedDerivative(4,0) = 4 * pow(0,4)/1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 2).matrix()).norm(), 0.0001);
	
	expectedDerivative(4,0) = 0 * pow(4,0) * log(4);
	expectedDerivative(3,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(2,0) = 2 * pow(2,2) * log(2);
	expectedDerivative(1,0) = 3 * pow(1,3) * log(1);
	expectedDerivative(0,0) = 4 * pow(0,4) * log(0); // Log by 0
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 3).matrix()).norm(), 0.0001);
	
	// COS NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeCos({1.0});
	
	expectedDerivative(0,0) = -1 * sin(0);
	expectedDerivative(1,0) = -1 * sin(1);
	expectedDerivative(2,0) = -1 * sin(2);
	expectedDerivative(3,0) = -1 * sin(3);
	expectedDerivative(4,0) = -1 * sin(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = -0 * sin(0);
	expectedDerivative(1,0) = -1 * sin(1);
	expectedDerivative(2,0) = -2 * sin(2);
	expectedDerivative(3,0) = -3 * sin(3);
	expectedDerivative(4,0) = -4 * sin(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// SIN NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeSin({1.0});
	
	expectedDerivative(0,0) = 1 * cos(0);
	expectedDerivative(1,0) = 1 * cos(1);
	expectedDerivative(2,0) = 1 * cos(2);
	expectedDerivative(3,0) = 1 * cos(3);
	expectedDerivative(4,0) = 1 * cos(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 0 * cos(0);
	expectedDerivative(1,0) = 1 * cos(1);
	expectedDerivative(2,0) = 2 * cos(2);
	expectedDerivative(3,0) = 3 * cos(3);
	expectedDerivative(4,0) = 4 * cos(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// ^3 NODE CHECK  -------------------------------------------------------------------------------
	toTest = new NodeCube({1.0});
	
	expectedDerivative(0,0) = 3 * pow(0,2);
	expectedDerivative(1,0) = 3 * pow(1,2);
	expectedDerivative(2,0) = 3 * pow(2,2);
	expectedDerivative(3,0) = 3 * pow(3,2);
	expectedDerivative(4,0) = 3 * pow(4,2);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 3 * 0;
	expectedDerivative(1,0) = 3 * 1;
	expectedDerivative(2,0) = 3 * 8;
	expectedDerivative(3,0) = 3 * 27;
	expectedDerivative(4,0) = 3 * 64;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// e^x NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeExponential({1.0});
	
	expectedDerivative(0,0) = 1 * exp(0);
	expectedDerivative(1,0) = 1 * exp(1);
	expectedDerivative(2,0) = 1 * exp(2);
	expectedDerivative(3,0) = 1 * exp(3);
	expectedDerivative(4,0) = 1 * exp(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);

	expectedDerivative(0,0) = 0 * exp(0);
	expectedDerivative(1,0) = 1 * exp(1);
	expectedDerivative(2,0) = 2 * exp(2);
	expectedDerivative(3,0) = 3 * exp(3);
	expectedDerivative(4,0) = 4 * exp(4);
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// GAUS NODE CHECK-------------------------------------------------------------------------------
	toTest = new NodeGaussian({1.0});
	
	expectedDerivative(0,0) = 2 * (1 - 0) * exp(-pow(1 - 0, 2));
	expectedDerivative(1,0) = 2 * (1 - 1) * exp(-pow(1 - 1, 2));
	expectedDerivative(2,0) = 2 * (1 - 2) * exp(-pow(1 - 2, 2));
	expectedDerivative(3,0) = 2 * (1 - 3) * exp(-pow(1 - 3, 2));
	expectedDerivative(4,0) = 2 * (1 - 4) * exp(-pow(1 - 4, 2));
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 2 * (0 - 1) * exp(-pow(1 - 0, 2));
	expectedDerivative(1,0) = 2 * (1 - 1) * exp(-pow(1 - 1, 2));
	expectedDerivative(2,0) = 2 * (2 - 1) * exp(-pow(1 - 2, 2));
	expectedDerivative(3,0) = 2 * (3 - 1) * exp(-pow(1 - 3, 2));
	expectedDerivative(4,0) = 2 * (4 - 1) * exp(-pow(1 - 4, 2));
	
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// LOG NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeLog({1.0});
	
	expectedDerivative(0,0) = MAX_FLT;
	expectedDerivative(1,0) = 1.0;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 1.0/4; // Check if this is intended
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// LOGIT NODE CHECK------------------------------------------------------------------------------
	toTest = new NodeLogit({1.0});
	
	expectedDerivative(0,0) = (1 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	expectedDerivative(1,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(2,0) = (1 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(4,0) = (1 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
    
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = (0 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	expectedDerivative(1,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(2,0) = (2 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (3 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(4,0) = (4 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
    
    ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
    
	// RELU NODE CHECK------------------------------------------------------------------------------
    toTest = new NodeRelu({1.0});
    
    expectedDerivative(0,0) = 0.01;
    expectedDerivative(1,0) = 1;
    expectedDerivative(2,0) = 1;
    expectedDerivative(3,0) = 1;
    expectedDerivative(4,0) = 1;
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
    expectedDerivative(0,0) = 0.01;
    expectedDerivative(1,0) = 1;
    expectedDerivative(2,0) = 2;
    expectedDerivative(3,0) = 3;
    expectedDerivative(4,0) = 4;
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// SQRT NODE CHECK-------------------------------------------------------------------------------
	toTest = new NodeSqrt({1.0});
	
	expectedDerivative(0,0) = 1/(2 * sqrt(0)); // divide by zero
	expectedDerivative(1,0) = 1/(2 * sqrt(1));
	expectedDerivative(2,0) = 1/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(3));
	expectedDerivative(4,0) = 1/(2 * sqrt(4)); 
    
    ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 0/(2 * sqrt(0)); //divide by zero
	expectedDerivative(1,0) = 1/(2 * sqrt(1));
	expectedDerivative(2,0) = 2/(2 * sqrt(2));
	expectedDerivative(3,0) = 3/(2 * sqrt(3));
	expectedDerivative(4,0) = 4/(2 * sqrt(4)); 
	
	ASSERT_LE((limited(expectedDerivative).matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
	// ^2  NODE CHECK -------------------------------------------------------------------------------
	toTest = new NodeSquare({1.0});
	
	expectedDerivative(0,0) = 2 * 1 * 0;
	expectedDerivative(1,0) = 2 * 1 * 1;
	expectedDerivative(2,0) = 2 * 1 * 2;
	expectedDerivative(3,0) = 2 * 1 * 3;
	expectedDerivative(4,0) = 2 * 1 * 4;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
    
	expectedDerivative(0,0) = 2 * 0;
	expectedDerivative(1,0) = 2 * 1;
	expectedDerivative(2,0) = 2 * 4;
	expectedDerivative(3,0) = 2 * 9;
	expectedDerivative(4,0) = 2 * 16;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);

	// TANH NODE CHECK-------------------------------------------------------------------------------
	toTest = new NodeTanh({1.0});
	
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 0.41997434161402606939449673904170;
	expectedDerivative(2,0) = 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 0.00986603716544019127315616968;
	expectedDerivative(4,0) = 0.0013409506830258968799702;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 0).matrix()).norm(), 0.0001);
	
	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1 * 0.41997434161402606939449673904170;
	expectedDerivative(2,0) = 2 * 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 3 * 0.00986603716544019127315616968;
	expectedDerivative(4,0) = 4 * 0.0013409506830258968799702;
	
	ASSERT_LE((expectedDerivative.matrix() - toTest->getDerivative(trace, 1).matrix()).norm(), 0.0001);
	
}

