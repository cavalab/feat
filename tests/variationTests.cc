#include "testsHeader.h"
#include "cudaTestUtils.h"

#ifndef USE_CUDA
bool isValidProgram(NodeVector& program, unsigned num_features)
{
    //checks whether program fulfills all its arities.
    MatrixXd X = MatrixXd::Zero(num_features,2); 
    VectorXd y = VectorXd::Zero(2);
    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z; 
    
    State state;
    Data data(X, y, z);
   
    for (const auto& n : program){
        if ( state.f.size() >= n->arity['f'] && state.b.size() >= n->arity['b'])
            n->evaluate(data, state);
        else
            return false; 
    }
    return true;
}
#endif

TEST(Variation, MutationTests)
{
	Feat feat(100, 100, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
	
    MatrixXd X(4,2); 
    MatrixXd X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z;

    VectorXd y(4); 
    VectorXd y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;

    feat.params.init();       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.p_ml = make_shared<ML>(feat.params); // intialize ML
    feat.p_pop = make_shared<Population>(feat.params.pop_size);
    feat.p_eval = make_shared<Evaluation>(feat.params.scorer);

	feat.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    
    Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv);
    
    feat.initial_model(d);
                  
    // initialize population 
    feat.p_pop->init(feat.best_ind, feat.params);
    
    feat.F.resize(X.cols(),int(2*feat.params.pop_size));
    
    feat.p_eval->fitness(feat.p_pop->individuals, dt,feat.F,feat.params);
    
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (feat.p_pop->size() < 2*feat.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
		int pass = feat.p_variation->mutate(feat.p_pop->individuals[mom],child,feat.params);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			// give child an open location in F
			child.loc = feat.p_pop->get_open_loc(); 
			//push child into pop
			feat.p_pop->individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < feat.p_pop->individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(feat.p_pop->individuals[i].program, feat.params.terminals.size()));
	
}

TEST(Variation, CrossoverTests)
{
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
	
    MatrixXd X(4,2); 
    MatrixXd X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    VectorXd y(4); 
    VectorXd y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;

    std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > z; 

    feat.params.init();       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.p_ml = make_shared<ML>(feat.params); // intialize ML
    feat.p_pop = make_shared<Population>(feat.params.pop_size);
    feat.p_eval = make_shared<Evaluation>(feat.params.scorer);
    
	feat.params.set_terminals(X.rows()); 
	
	Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv);
        
    // initial model on raw input
    feat.initial_model(d);

                  
    // initialize population 
    feat.p_pop->init(feat.best_ind, feat.params);
    
    feat.F.resize(X.cols(),int(2*feat.params.pop_size));
    
    feat.p_eval->fitness(feat.p_pop->individuals,dt,feat.F,feat.params);
    
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (feat.p_pop->size() < 2*feat.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
        int dad = r.random_choice(parents);
        
		int pass = feat.p_variation->cross(feat.p_pop->individuals[mom],feat.p_pop->individuals[dad],child,feat.params);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			// give child an open location in F
			child.loc = feat.p_pop->get_open_loc(); 
			//push child into pop
			feat.p_pop->individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < feat.p_pop->individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(feat.p_pop->individuals[i].program, feat.params.terminals.size()));
}

