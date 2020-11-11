#include "testsHeader.h"
#include "cudaTestUtils.h"

#ifndef USE_CUDA
bool isValidProgram(NodeVector& program, unsigned num_features)
{
    //checks whether program fulfills all its arities.
    MatrixXf X = MatrixXf::Zero(num_features,2); 
    VectorXf y = VectorXf::Zero(2);
    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 
    
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
	
    MatrixXf X(4,2); 
    MatrixXf X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z;

    VectorXf y(4); 
    VectorXf y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;

    feat.params.init(X, y);       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.pop = Population(feat.params.pop_size);
    feat.evaluator = Evaluation(feat.params.scorer_);

	feat.params.set_terminals(X.rows()); 
        
    // initial model on raw input
    
    Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv);
    
    feat.initial_model(d);
                  
    // initialize population 
    feat.pop.init(feat.best_ind, feat.params);
    
    feat.evaluator.fitness(feat.pop.individuals, dt, feat.params);
    
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (feat.pop.size() < 2*feat.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
		int pass = feat.variator.mutate(feat.pop.individuals[mom],
                child,feat.params,dt);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			//push child into pop
			feat.pop.individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < feat.pop.individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(feat.pop.individuals[i].program, feat.params.terminals.size()));
	
}

TEST(Variation, CrossoverTests)
{
    Feat feat(100, 100, "LinearRidgeRegression", false, 1);
	feat.set_random_state(666);
	
    MatrixXf X(4,2); 
    MatrixXf X_v(3,2); 
    X << 0,1,  
         0.47942554,0.87758256,  
         0.84147098,  0.54030231,
         0.99749499,  0.0707372;
    X_v <<  0.90929743, -0.41614684,
         0.59847214, -0.80114362,
         0.14112001,-0.9899925;

    X.transposeInPlace();
    X_v.transposeInPlace();

    VectorXf y(4); 
    VectorXf y_v(3); 
    // y = 2*x1 + 3.x2
    y << 3.0,  3.59159876,  3.30384889,  2.20720158;
    y_v << 0.57015434, -1.20648656, -2.68773747;

    std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > z; 

    feat.params.init(X, y);       
  
    feat.set_dtypes(find_dtypes(X));
            
    feat.pop = Population(feat.params.pop_size);
    feat.evaluator = Evaluation(feat.params.scorer_);
    
	feat.params.set_terminals(X.rows()); 
	
	Data dt(X, y, z);
    Data dv(X_v, y_v, z);
    
    DataRef d;
    
    d.setTrainingData(&dt);
    d.setValidationData(&dv);
        
    // initial model on raw input
    feat.initial_model(d);

                  
    // initialize population 
    feat.pop.init(feat.best_ind, feat.params);
    
    feat.evaluator.fitness(feat.pop.individuals,dt,feat.params);
    
    vector<size_t> parents;
    parents.push_back(2);
    parents.push_back(5);
    parents.push_back(7);
    
    while (feat.pop.size() < 2*feat.params.pop_size)
    {
		Individual child;
		
		int mom = r.random_choice(parents);
        int dad = r.random_choice(parents);
        
		int pass = feat.variator.cross(feat.pop.individuals[mom],
                                           feat.pop.individuals[dad],
                                           child,
                                           feat.params,*d.t);
		
		if (pass)                   // congrats! you produced a viable child.
		{
			//push child into pop
			feat.pop.individuals.push_back(child);
		}
	}
	
	int i;
	
	for(i = 0; i < feat.pop.individuals.size(); i++)
		ASSERT_TRUE(isValidProgram(feat.pop.individuals[i].program, 
                    feat.params.terminals.size()));
}

