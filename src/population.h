/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef POPULATION_H
#define POPULATION_H

using std::vector;
using Eigen::ArrayXd;
using Eigen::ArrayXi;

namespace FT{    
    struct Individual{
        //Represents individual programs in the population 
        
        Individual(){}

        ~Individual(){}
        
    };

    struct Population
    {
        vector<Individual> programs;

        Population(){}
        ~Population(){}

        void init(const Parameters& params)
        {
            // initialize population of programs. 
        }
        void update(vector<size_t> survivors)
        {
            // reduce programs to the indices in survivors.
        }
    };

    struct node
    {
        // represents nodes in a program. 
        
        char name;              // node type
        char otype;             // output type
        unsigned int arity_f;   // floating arity of the operator 
        unsigned int arity_b;   // floating arity of the operator 
        double value;           // value, for k and x types
        size_t loc;             // column location in X, for x types

        node(){}
        ~node(){}

        void evaluate(MatrixXd& X, vector<ArrayXd>& stack_f, vector<ArrayXi>& stack_b)
        {
            // evaluates a node and updates the stack state. 
                
            // check node arity 
            if (stack_f.size() >= arity_f && stack_b.size() >= arity_b)
            { 
                switch(name)
                {
                    case 'k': // push array of values equal to k
                    {   
                        if (otype == 'b')
                            stack_b.push_back(ArrayXi::Constant(X.rows(),int(value)));
                        else 	
                            stack_f.push_back(ArrayXd::Constant(X.rows(),value));
                        break;
                    }
                    case 'x': // push variable to correct stack
                    {
                        if (otype == 'b')
                            stack_b.push_back(X.col(loc).cast<int>());
                        else
                            stack_f.push_back(X.col(loc));
                        break;
                    }
                    case '+': // add  
                    {
                        ArrayXd x = stack_f.back(); stack_f.pop_back();
                        ArrayXd y = stack_f.back(); stack_f.pop_back();
                        stack_f.push_back(x + y);
                        break;
                    }
                    case '-': // subtract
                    {
                        ArrayXd x = stack_f.back(); stack_f.pop_back();
                        ArrayXd y = stack_f.back(); stack_f.pop_back();
                        stack_f.push_back(x - y);
                        break;
                    }
                    case '*': // multiply
                    {
                        ArrayXd x = stack_f.back(); stack_f.pop_back();
                        ArrayXd y = stack_f.back(); stack_f.pop_back();
                        stack_f.push_back(x * y);
                        break;
                    }
                    case '/': //divide
                    {
                        ArrayXd x = stack_f.back(); stack_f.pop_back();
                        ArrayXd y = stack_f.back(); stack_f.pop_back();
                        stack_f.push_back(x / y);
                        break;
                    }
                    case 'e': //exponential
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(exp(x));
                        break;
                    }
                    case 'l': //log
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(log(x));
                        break;
                    }
                    case 's': //sin
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(sin(x));
                        break;
                    }
                    case 'c': //cos
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(cos(x));
                        break;
                    }
                    case '2': //square
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(pow(x,2));
                        break;
                    }
                    case '3': //cube
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(pow(x,3));
                        break;
                    }
                    case 'q': //square root
                    {
                        ArrayXd x = stack_f.back(); stack_f.back();
                        stack_f.push_back(sqrt(abs(x)));
                        break;
                    }
                    case '^': //exponent 
                    {
                        ArrayXd x = stack_f.back(); stack_f.pop_back();
                        ArrayXd y = stack_f.back(); stack_f.pop_back();
                        stack_f.push_back(pow(x,y));
                        break;
                    }
                    default:
                    {
                        std::cerr << "invalid operator name\n";
                        break;
                    }
                }
            }
         }

    };
}
#endif
