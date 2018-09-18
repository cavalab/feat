#include "testsHeader.h"

TEST(Parameters, ParamsTests)
{                   
		Parameters params(100, 								//pop_size
					  100,								//gens
					  "LinearRidgeRegression",			//ml
					  false,							//classification
					  0,								//max_stall
					  'f',								//otype
					  2,								//verbosity
					  "+,-,*,/,exp,log",				//functions
					  0.5,                              //cross_rate
					  3,								//max_depth
					  10,								//max_dim
					  false,							//erc
					  "fitness,complexity",  			//obj
                      false,                            //shuffle
                      0.75,								//train/test split
                      0.5,                             // feedback 
                      "mse",                           //scoring function
                      "",                              // feature names
                      false,                            // backprop
                      0,                                // backprop iterations
                      0.1,                              // iterations
                      1,                                // batch size
                      false,                             // hill climbing
                      -1,                                // max time
                      false);                            // use batch training
					  
	params.set_max_dim(12);
	ASSERT_EQ(params.max_dim, 12);
	ASSERT_EQ(params.max_size, (pow(2,params.max_depth+1)-1)*params.max_dim);
	
	params.set_max_depth(4);
	ASSERT_EQ(params.max_depth, 4);
	ASSERT_EQ(params.max_size, (pow(2,params.max_depth+1)-1)*params.max_dim);
	
	params.set_terminals(10);
	
	ASSERT_EQ(params.terminals.size(), 10);
	
	vector<double> weights;
	weights.resize(10);
	weights[0] = 10;
	ASSERT_NO_THROW(params.set_term_weights(weights)); //TODO should throw error here as vector is empty just 
	
	params.set_functions("+,-,*,/");
	ASSERT_EQ(params.functions.size(), 4);
	
	int i;
	vector<string> function_names = {"+", "-", "*", "/"};
	for(i = 0; i < 4; i ++)
		ASSERT_STREQ(function_names[i].c_str(), params.functions[i]->name.c_str());
		
	params.set_objectives("fitness,complexity");
	ASSERT_EQ(params.objectives.size(), 2);
	
	ASSERT_STREQ(params.objectives[0].c_str(), "fitness");
	ASSERT_STREQ(params.objectives[1].c_str(), "complexity");
	
	params.set_verbosity(-1);
	ASSERT_EQ(params.verbosity, 2);
	
	params.set_verbosity(10);
	ASSERT_EQ(params.verbosity, 2);
	
	params.set_verbosity(3);
	ASSERT_EQ(params.verbosity, 3);
	
	string str1 = "Hello\n";
	string str2 = params.msg("Hello", 0);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = params.msg("Hello", 2);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = params.msg("Hello", 3);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	params.set_verbosity(2);
	ASSERT_EQ(params.verbosity, 2);
	ASSERT_STREQ("", params.msg("Hello", 3).c_str());
}

