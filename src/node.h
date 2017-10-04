/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_H
#define NODE_H

using std::vector;
using std::string;
using Eigen::ArrayXd;
using Eigen::ArrayXi;


namespace FT{

    //////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Node
     * @brief represents nodes in a program. 
     */
    class Node
    {        
        public:
            char name;              ///< node type
            char otype;             ///< output type
            unsigned int arity_f;   ///< floating arity of the operator 
            unsigned int arity_b;   ///< boolean arity of the operator 
            double value;           ///< value, for k and x types
            size_t loc;             ///< column location in X, for x types
            int complexity;         ///< complexity of node

            // constructors 
            /*!
             * @brief node representing an operator.
             */
            Node(char n);
            /*!
             * @brief node representing a variable terminal.
             */
            Node(char n, const size_t& l);
            /*!
             * @brief node representing a floating point value terminal.
             */
            Node(char n, const double& v);
            
            ~Node(){}
            
            /*!
             * @brief evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                          vector<ArrayXi>& stack_b);

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b);

            /*!
             * @brief total arity
             */
            unsigned int total_arity(){ return arity_f + arity_b; }

        private:
        	/*!
             * @brief  disallows constructors that don't at least specify a node type
             */
            Node(){}                 
                                    
            /*!
             * sets complexity of operator
             */
            void set_complexity();
    };
    
    ///////////////////////////////////////////////////////////////////////////////// Definitions
    
    //evaluates a node and updates the stack state
    void Node::evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                      vector<ArrayXi>& stack_b)
    {
                           
        // check node arity 
        if (stack_f.size() >= arity_f && stack_b.size() >= arity_b)
        { 
            switch(name)
            {
                case 'k': // push array of values equal to k
                {   
                    if (otype == 'b')
                        stack_b.push_back(ArrayXi::Constant(X.cols(),int(value)));
                    else 	
                        stack_f.push_back(ArrayXd::Constant(X.cols(),value));
                    break;
                }
                case 'x': // push variable to correct stack
                {
                    if (otype == 'b')
                        stack_b.push_back(X.row(loc).cast<int>());
                    else
                        stack_f.push_back(X.row(loc));
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
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(exp(x));
                    break;
                }
                case 'l': //log
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(log(x));
                    break;
                }
                case 's': //sin
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(sin(x));
                    break;
                }
                case 'c': //cos
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(cos(x));
                    break;
                }
                case '2': //square
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(pow(x,2));
                    break;
                }
                case '3': //cube
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(pow(x,3));
                    break;
                }
                case 'q': //square root
                {
                    ArrayXd x = stack_f.back(); stack_f.pop_back();
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

    // evaluates the node symbolically.
    void Node::eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
    {

        // check node arity 
        if (stack_f.size() >= arity_f && stack_b.size() >= arity_b)
        { 
            switch(name)
            {
                case 'k': // push value k
                {   
                    if (otype == 'b')
                        stack_b.push_back(std::to_string(value));
                    else 	
                        stack_f.push_back(std::to_string(value));
                    break;
                }
                case 'x': // push variable to correct stack
                {
                    if (otype == 'b')
                        stack_b.push_back("x_" + std::to_string(loc));
                    else
                        stack_f.push_back("x_" + std::to_string(loc));
                    break;
                }
                case '+': // add  
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    string y = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "+" + y + ")");
                    break;
                }
                case '-': // subtract
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    string y = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "-" + y + ")");
                    break;
                }
                case '*': // multiply
                {
                   string x = stack_f.back(); stack_f.pop_back();
                   string y = stack_f.back(); stack_f.pop_back();
                   stack_f.push_back("(" + x + "*" + y + ")");
                   break;
                }
                case '/': //divide
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    string y = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "/" + y + ")");
                    break;
                }
                case 'e': //exponential
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("exp(" + x + ")");
                    break;
                }
                case 'l': //log
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("log(" + x + ")");
                    break;
                }
                case 's': //sin
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("sin(" + x + ")");
                    break;
                }
                case 'c': //cos
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("cos(" + x + ")");
                    break;
                }
                case '2': //square
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "^2)");
                    break;
                }
                case '3': //cube
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "^3)");
                    break;
                }
                case 'q': //square root
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("sqrt(|" + x + "|)");
                    break;
                }
                case '^': //exponent 
                {
                    string x = stack_f.back(); stack_f.pop_back();
                    string y = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + ")^(" + y + ")");
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
    
    Node::Node(char n)
    {
        /* node representing an operator. */
        name = n;

 	    if (name=='s' || name=='c' || name=='e' || name=='l' || name=='q')
        {
			arity_f = 1;
			arity_b=0;
			otype='f';
		}
		else if (name=='+' || name=='-' || name=='*' || name=='/' || name=='^')
        {
			arity_f=2;
			arity_b=0;
			otype='f';
		}
		else if (name=='<' || name=='>' || name=='{' || name=='}' || name=='=')
        {
			arity_f=2;
			arity_b=0;
			otype='b';
		}
		else if (name=='&' || name=='|' )
        {
			arity_f = 0;
			arity_b = 2;
			otype='b';
		}
		else if (name=='!')
        {
			arity_f=0;
			arity_b=1;
			otype='b';
		}
		else if (name=='i')
        {
			arity_b=1;
			arity_f=1;
			otype='f';
		}
		else if (name=='t')
        {
			arity_b=1;
			arity_f=2;
			otype='f';
		}
		else{
			std::cerr << "error in node.h: node name not specified";
			throw;
		}
		
		// assign complexity
		set_complexity(); 
         
    }
    Node::Node(char n, const size_t& l)
    {
        /* node representing a variable terminal.*/
        name = n;
        loc = l; 
        arity_f = 0;
        arity_b = 0; 
        otype = 'f';
        complexity = 1;
    }
    Node::Node(char n, const double& v)
    {
        /* node representing a floating point value terminal.*/
        name = n;
        value = v;
        arity_f = 0;
        arity_b = 0;
        otype = 'f';
        complexity = 1;

    }
    void Node::set_complexity()
    {
        /* assign complexity to nodes.*/
		
		if (name=='i' || name=='t')
			complexity = 5;
		else if (name=='e' || name=='l' || name=='^')
			complexity = 4;
		else if (name=='s' || name=='c' )
			complexity = 3;
		else if (name=='/' || name=='q' || name=='<' || name=='>' || name=='{' || name=='}' 
                || name=='&' || name=='|')
			complexity = 2;
		else
			complexity = 1;
    }
}
#endif
