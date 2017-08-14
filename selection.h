/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

struct Selection
{
    // implements selection methods. 
    SelectionOperator * selector; 
    
    Selection(string type, bool survival=false)
    {
        // set type of selection operator. 
        if (!type.compare("lexicase"))
            selector = new Lexicase(survival); 
        else if (!type.compare("pareto"))
            selector = new Pareto(survival);
            
    }
    ~Selection(){
        delete selector; 
    }

    // perform selection
    vector<size_t> choices = select(const MatrixXd& F, const int pop_size)
    {       
        return selector->select(const MatrixXd& F, const int pop_size);
    }
};


struct SelectionOperator 
{
    // base class for selection operators. 
    SelectionOperator(){}
    ~SelectionOperator(){}
    vector<size_t> choices = select(const MatrixXd& F, const int pop_size){}
    
};

struct Lexicase : SelectionOperator
{
    // Lexicase selection operator.
    vector<size_t> choices = select(){}

};

struct Tournament: SelectionOperator
{
    // tournament selection operator.
};

struct Pareto : SelectionOperator
{
    // Pareto selection operator.
};
