#include "testsHeader.h"

TEST(Population, PopulationTests)
{
	Feat feat(100, 100, "LinearRidgeRegression", false, 1);
    feat.set_n_threads(1);
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

    feat.params.init();       
  
    feat.set_dtypes(find_dtypes(X));
    cout << "dtypes: ";
    for (const auto& d : feat.params.dtypes)
        cout << d << ",";
    cout << "\n";
    feat.p_ml = make_shared<ML>(); // intialize ML
    feat.p_pop = make_shared<Population>(feat.params.pop_size);  
    //std::cout << "feat.params.scorer: " << feat.params.scorer << "\n";
    feat.p_eval = make_shared<Evaluation>("mse");

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
    
    vector<size_t> current_locs; 
    int i;
    for(i = 0; i < feat.p_pop->individuals.size(); i++)
    {
    	ASSERT_NO_THROW(feat.p_pop->individuals[i]);
	    current_locs.push_back(feat.p_pop->individuals[i].loc);
	}
	    
	size_t loc1 = feat.p_pop->get_open_loc();
	size_t loc2 = feat.p_pop->get_open_loc();
	size_t loc3 = feat.p_pop->get_open_loc();
	size_t loc4 = feat.p_pop->get_open_loc();
	
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc1) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc2) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc3) != current_locs.end());
	ASSERT_FALSE(std::find(current_locs.begin(), current_locs.end(), loc4) != current_locs.end());
	
	//TODO put a check in get_open_loc()
}

