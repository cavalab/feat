#pragma once
#ifndef POPULATION_H
#define POPULATION_H

using namespace std;

struct Individual{
    //Represents individual programs in the populatio 
    
    Individual(){}

    ~Individual(){}
    
};

/*void Fewtwo::init_pop(){
    // initializes population of programs.
    //
}
*/
struct Population
{
    vector<Individual> programs;

    Population(){}
    ~Population(){}

    void init(int pop_size)
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
    
    char name; // node type
    char otype; //output type
    unsigned int arity_f; // floating arity of the operator 
    unsigned int arity_b; // floating arity of the operator 
    double value;

    node(){}
    ~node(){}

    void eval(MatrixXf& X, vector<ArrayXf>& stack_f, vector<ArrayXb>& stack_b)
    {
        // evaluates a node and updates the stack state. 
            
        // check node arity 
        if (stack_f.size() >= arity_f && stack_b.size() >= arity_b)
        { 
            switch(name)
            {
            case 'k': // push array of values equal to k
                if (otype == 'b'):
                    stack_b.push_back(ArrayXb(value));
                else 	
                    stack_f.push_back(ArrayXf(value));
                break;
            case 'x': // push variable to correct stack
                if (otype == 'b'):
                    stack_b.push_back(X.col(loc));
                else
                    stack_f.push_back(X.col(loc));
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

}
#endif
