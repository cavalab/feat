#include "testsHeader.h"

TEST(Parameters, paramsTests)
{                   
    Feat ft;
    					  
	ft.params.set_max_dim(12);
	ASSERT_EQ(ft.params.max_dim, 12);
	ASSERT_EQ(ft.params.max_size, (pow(2,ft.params.max_depth+1)-1)*ft.params.max_dim);
	
	ft.params.set_max_depth(4);
	ASSERT_EQ(ft.params.max_depth, 4);
	ASSERT_EQ(ft.params.max_size, (pow(2,ft.params.max_depth+1)-1)*ft.params.max_dim);
	
	ft.params.set_terminals(10);
	
	ASSERT_EQ(ft.params.terminals.size(), 10);
	
	vector<float> weights;
	weights.resize(10);
	weights[0] = 10;
	ASSERT_NO_THROW(ft.params.set_term_weights(weights)); //TODO should throw error here as vector is empty just 
	
	ft.params.set_functions("+,-,*,/");
	ASSERT_EQ(ft.params.functions.size(), 4);
	
	int i;
	vector<string> function_names = {"+", "-", "*", "/"};
	for(i = 0; i < 4; i ++)
		ASSERT_STREQ(function_names[i].c_str(), ft.params.functions[i]->name.c_str());
		
	ft.params.set_objectives("fitness,complexity");
	ASSERT_EQ(ft.params.objectives.size(), 2);
	
	ASSERT_STREQ(ft.params.objectives[0].c_str(), "fitness");
	ASSERT_STREQ(ft.params.objectives[1].c_str(), "complexity");
	
	ft.params.set_verbosity(-1);
	ASSERT_EQ(ft.params.verbosity, 2);
	
	ft.params.set_verbosity(10);
	ASSERT_EQ(ft.params.verbosity, 2);
	
	ft.params.set_verbosity(3);
	ASSERT_EQ(ft.params.verbosity, 3);
	
	string str1 = "Hello\n";
	string str2 = logger.log("Hello", 0);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = logger.log("Hello", 2);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	str2 = logger.log("Hello", 3);
	ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	ft.params.set_verbosity(2);
	ASSERT_EQ(ft.params.verbosity, 2);
	ASSERT_STREQ("", logger.log("Hello", 3).c_str());
}

