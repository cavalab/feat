// William La Cava 2017
#include "population.h"
#include "rnd.h"
#include <Eigen/Dense.h>
using namespace Eigen;
using namespace std;
using Eigen::MatrixXf;
using Eigen::ArrayXf;

// code to evaluate GP programs.

void eval(node& n, MatrixXf& X, vector<ArrayXf>& stack_f, vector<ArrayXb>& stack_b)
{
	// evaluates a node and updates the stack state. 
	
	// check node arity 
	if (stack_f.size() >= n.arity_f && stack_b.size() >= n.arity_b)
	{ 
		switch(n.name)
		{
		case 'k': // push array of values equal to k
			if (n.otype == 'b'):
				stack_b.push_back(ArrayXb(n.value));
			else 	
				stack_f.push_back(ArrayXf(n.value));
			break;
		case 'x': // push variable to correct stack
			if (n.otype == 'b'):
				stack_b.push_back(X.col(n.loc));
			else
				stack_f.push_back(X.col(n.loc));
			break;
		case '+': // add  
			ArrayXf x = stack_f.back(); stack_f.pop_back();
			ArrayXf y = stack_f.back(); stack_f.pop_back();
			stack_f.push_back(x + y);
			break;
		case '-': // subtract
			ArrayXf x = stack_f.back(); stack_f.pop_back();
			ArrayXf y = stack_f.back(); stack_f.pop_back();
			stack_f.push_back(x - y);
			break;
		case '*': // multiply
			ArrayXf x = stack_f.back(); stack_f.pop_back();
			ArrayXf y = stack_f.back(); stack_f.pop_back();
			stack_f.push_back(x * y);
			break;
		case '/': //divide
			ArrayXf x = stack_f.back(); stack_f.pop_back();
			ArrayXf y = stack_f.back(); stack_f.pop_back();
			stack_f.push_back(x / y);
			break;
		case 'e': //exponential
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(exp(x));
			break;
		case 'l': //log
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(log(x));
			break;
		case 's': //sin
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(sin(x));
			break;
		case 'c': //cos
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(cos(x));
			break;
		case '2': //square
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(pow(x,2));
			break;
		case '3': //cube
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(pow(x,3));
			break;
		case 'q': //square root
			ArrayXf x = stack_f.back(); stack_f.back();
			stack_f.push_back(sqrt(abs(x)));
			break;
		case '^': //exponent 
			ArrayXf x = stack_f.back(); stack_f.pop_back();
			ArrayXf y = stack_f.back(); stack_f.pop_back();
			stack_f.push_back(pow(x,y));
			break;
		default:
			err << "invalid operator name\n";
			break;
		}
	}
}


