// William La Cava 2017
#include "population.h"
#include "rnd.h"
#include <Eigen/Dense.h>
using namespace Eigen;
using namespace std;
using Eigen::MatrixXf;
using Eigen::ArrayXf;

// code to evaluate GP programs.

void eval(node& n, vector<ArrayXf>& stack_f, vector<ArrayXb>& stack_b)
{
	if (stack_f.size() >= n.arity_f && stack_b.size() >= n.arity_b)
	{
		switch(n.name)
		{
		case 'k': // push array of values equal to k
			if (k.otype == 'b'):
				stack_b.push_back(ArrayXb(n.value));
			else 	
				stack_f.push_back(ArrayXf(n.value));
		case 'x': // push variable to correct stack
		case '+': // add values in correct stack
		case '-':
		case '*':
		case '/':
		case 'e':
		case 'l':
		case 's':
		case 'c':
		case '2':
		case '3':
		case 'q':
		case ''
		}
	}
}


