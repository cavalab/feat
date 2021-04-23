/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "variation.h"

namespace FT{
namespace Vary{
/// constructor
Variation::Variation(float cr): cross_rate(cr) {}
           
/// update cross rate
void Variation::set_cross_rate(float cr)
{
    cross_rate = cr;
}

/// return current cross rate
float Variation::get_cross_rate()
{
    return cross_rate;
}

 /// destructor
Variation::~Variation(){}

std::unique_ptr<Node> random_node(const NodeVector & v)
{
   /*!
    * return a random node from a list of nodes.
    */          
    assert(v.size()>0 && " attemping to return random choice from empty vector");
    std::vector<size_t> vi(v.size());
    std::iota(vi.begin(), vi.end(), 0);
    size_t idx = r.random_choice(vi);
    return v.at(idx)->clone();
}

void Variation::vary(Population& pop, const vector<size_t>& parents, 
                     const Parameters& params, const Data& d)
{
    /*!
     * performs variation on the current population. 
     *
     * @param   pop: current population
     * @param  	parents: indices of population to use for variation
     * @param  	params: feat parameters
     *
     * @return  appends params.pop_size offspring derived from parent variation
     */
    unsigned start= pop.size();
    pop.resize(2*params.pop_size);
    #pragma omp parallel for
    for (unsigned i = start; i<pop.size(); ++i)
    {
        // pass check for children undergoing variation     
        bool pass=false;                     
        while (!pass)
        {
            Individual child; // new individual
            child.set_id(params.current_gen*params.pop_size+i-start);           

            if ( r() < cross_rate)      // crossover
            {
                // get random mom and dad, make copies 
                Individual& mom = pop.individuals.at(r.random_choice(parents));
                Individual& dad = pop.individuals.at(r.random_choice(parents));
                /* int dad = r.random_choice(parents); */
                // create child
               
                // perform crossover
                logger.log("\n===\ncrossing\n" + mom.get_eqn() + "\nwith\n " + 
                           dad.get_eqn() , 3);
                logger.log("programs:\n" + mom.program_str() + "\nwith\n " + 
                           dad.program_str() , 3);
                
                pass = cross(mom, dad, child, params, d);
                
                logger.log("crossing " + mom.get_eqn() + "\nwith\n " + 
                   dad.get_eqn() + "\nproduced " + child.get_eqn() + 
                   ", pass: " + std::to_string(pass) + "\n===\n",3);    
                
                child.set_parents({mom, dad});
            }
            else                        // mutation
            {
                // get random mom
                Individual& mom = pop.individuals.at(r.random_choice(parents));
                /* int mom = r.random_choice(parents); */                
                // create child
                /* #pragma omp critical */
                /* { */
               logger.log("mutating " + mom.get_eqn() + "(" + 
                        mom.program_str() + ")", 3);
               pass = mutate(mom,child,params,d);
               logger.log("mutating " + mom.get_eqn() + " produced " + 
                        child.get_eqn() + ", pass: " + std::to_string(pass),3);
                /* } */ 
                child.set_parents({mom});
            }
            if (pass)
            {
                assert(child.size()>0);
                logger.log("assigning " + child.program_str() + 
                        " to pop.individuals[" + std::to_string(i) + "]",3);

                pop.individuals.at(i) = child;
            }
        }    
   }
}

bool Variation::mutate(const Individual& mom, Individual& child, 
        const Parameters& params, const Data& d)
{
    /*!
     * chooses uniformly between point, insert and delete mutation 
     * 
     * @param   mom: parent
     * @param   child: offspring produced by mutating mom 
     * @param   params: parameters
     * 
     * @return  true if valid child, false if not 
     */    

    // make child a copy of mom
    mom.clone(child, false);  
    float rf = r();
    if (rf < 1.0/3.0 && child.get_dim() > 1)
    {
        if (r() < 0.5)
        {
            logger.log("\tdeletion mutation",3);
            delete_mutate(child,params); 
        }
        else 
        {
            if (params.corr_delete_mutate)
            {
                logger.log("\tcorrelation_delete_mutate",3);
                bool perfect_correlation = correlation_delete_mutate(
                        child,mom.Phi,params,d); 
            }
            else
            {
                logger.log("\tdelete_dimension_mutate",3);
                delete_dimension_mutate(child, params);
            }
        }
        assert(child.program.is_valid_program(params.num_features, 
                    params.longitudinalMap));
    }
    else if (rf < 2.0/3.0 && child.size() < params.max_size)
    {
        logger.log("\tinsert mutation",3);
        insert_mutate(child,params);
        assert(child.program.is_valid_program(params.num_features, 
                    params.longitudinalMap));
    }
    else
    {        
        logger.log("\tpoint mutation",3);
        point_mutate(child,params);
        assert(child.program.is_valid_program(params.num_features, 
                    params.longitudinalMap));
    }

    // check child depth and dimensionality
    return child.size()>0 && child.size() <= params.max_size 
            && child.get_dim() <= params.max_dim;
}

void Variation::point_mutate(Individual& child, const Parameters& params)
{
    /*! 1/n point mutation. 
     * @param child: individual to be mutated
     * @param params: parameters 
     * @return modified child
     * */
    float n = child.size(); 
    unsigned i = 0;
    // loop thru child's program
    for (auto& p : child.program)
    {
        /* cout << "child.get_p(i): "; */
        /* cout << child.get_p(i) << "\n"; */
        if (r() < child.get_p(i))  // mutate p. 
        {
            logger.log("\t\tmutating node " + p->name, 3);
            NodeVector replacements;  // potential replacements for p

            if (p->total_arity() > 0) // then it is an instruction
            {
                // find instructions with matching in/out types and arities
                for (const auto& f: params.functions)
                {
                    if (f->otype == p->otype &&
                            f->arity == p->arity)
                        /* f->arity.at('f')==p->arity.at('f') && */ 
                        /* f->arity.at('b')==p->arity.at('b') && */
                        /* f->arity.at('c')==p->arity.at('c') && */
                        /* f->arity.at('z')==p->arity.at('z')) */
                        replacements.push_back(f->rnd_clone());
                }
            }
            else                    // otherwise it is a terminal
            {
                // TODO: add terminal weights here
                // find terminals with matching output types
                                  
                for (const auto& t : params.terminals)
                {
                    if (t->otype == p->otype)
                        replacements.push_back(t->clone());
                }                                       
            }
            // replace p with a random one
            if (replacements.size() == 0)
            {
                WARN("WARNING: couldn't mutate " + 
                        to_string(p->name)+ ", no valid replacements found\n");
            }
            else
                p = random_node(replacements);  
        }
        ++i; 
    }

}

void Variation::insert_mutate(Individual& child, 
        const Parameters& params)
{        
    /*! insertion mutation. 
     * @param child: indiviudal to be mutated
     * @param params: parameters
     * @return modified child
     * */
    
    float n = child.size(); 
    
    if (r()<0.5 || child.get_dim() == params.max_dim)
    {
        // loop thru child's program
        for (unsigned i = 0; i< child.program.size(); ++i)
        {
            // mutate with weighted probability
            if (r() < child.get_p(i))                      
            {
                logger.log("\t\tinsert mutating node " + 
                        child.program.at(i)->name + " with probability " +
                        std::to_string(child.get_p(i)), 3);
                NodeVector insertion;  // inserted segment
                NodeVector fns;  // potential fns 
                
                // find instructions with matching output types and a 
                // matching arity to i
                for (const auto& f: params.functions)
                { 
                    // find fns with matching output types that take 
                    // this node type as arg
                    if (f->otype==child.program.at(i)->otype 
                        && f->arity.at(child.program.at(i)->otype) > 0)
                    { 
                        // make sure there are satisfactory types in 
                        // terminals to fill fns' args
                        map<char,unsigned> fn_arity = f->arity;
                        --fn_arity.at(child.program.at(i)->otype);
                        bool valid_function = true;
                        for (auto kv : fn_arity)
                        {
                            if (kv.second > 0)
                            {
                                if (!in(params.dtypes, kv.first))
                                {
                                    valid_function=false;
                                }
                            }
                            
                        }
                        if (valid_function)
                            fns.push_back(f->rnd_clone());
                    }
                }

                if (fns.size()==0)  // if no insertion functions match, skip
                    continue;

                // grab chosen node's subtree
                int end = i;
                int start = child.program.subtree(end); 
                logger.log("\t\tinsert mutation from " + to_string(end)
                        + " to " + to_string(start), 3);


                // choose a function to insert                    
                insertion.push_back(random_node(fns));
                // now we need to manually construct a subtree with this 
                // insertion node, with the child program stiched in as an 
                // argument.
                map<char, unsigned> insert_arity = insertion.back()->arity;

                // decrement function arity by one for child node 
                --insert_arity.at(child.program.at(i)->otype);
                 
                vector<char> type_order = {'f','b','c','z'};
                for (auto type : type_order)
                {
                    // add the child now if type matches
                    if (child.program.at(i)->otype==type)                        
                    {
                        /* cout << "adding " << type << " child program: [" ; */ 
                        for (int k = end; k != start-1; --k)
                        {
                            /* cout << child.program.at(k)->name ; */
                            /* cout << " (k=" << k << ")\t"; */
                            insertion.push_back(child.program.at(k)->clone());
                        }
                        /* cout << "]\n"; */
                    }
                    // push back new arguments for the rest of the function
                    for (unsigned j = 0; j< insert_arity.at(type); ++j)
                    {
                        insertion.make_tree(params.functions,params.terminals,
                                0, params.term_weights,params.op_weights,
                                type, params.ttypes);
                    }
                }
                // post-fix notation
                std::reverse(insertion.begin(),insertion.end());
               
                string s; 
                for (const auto& ins : insertion) s += ins->name + " "; 
                logger.log("\t\tinsertion: " + s + "\n", 3);
                NodeVector new_program; 
                splice_programs(new_program, 
                                child.program, start, end, 
                                insertion, size_t(0), insertion.size()-1);
                child.program=new_program;
                i += insertion.size()-1;
            }
            /* std::cout << "i: " << i << "\n"; */ 
        }
        /* cout << child.get_eqn() << "\n"; */
    }
    else    // add a dimension
    {            
        NodeVector insertion; // new dimension
        insertion.make_program(params.functions, params.terminals, 1,  
                     params.term_weights,params.op_weights,1,
                     r.random_choice(params.otypes), params.longitudinalMap, 
                     params.ttypes);
        for (const auto& ip : insertion) 
            child.program.push_back(ip->clone());
    }
}

void Variation::delete_mutate(Individual& child, 
        const Parameters& params)
{
    /*! deletion mutation. works by pruning a dimension. 
     * @param child: individual to be mutated
     * @param params: parameters  
     * @return mutated child
     * */
    logger.log("\t\tprogram: " + child.program_str(),3);
    // loop thru child's program
    for (unsigned i = 0; i< child.program.size(); ++i)
    {
        // mutate with weighted probability
        if (child.program.subtree(i) != i && r() < child.get_p(i))                      
        {
            // get subtree indices of program to delete
            size_t end = i; 
            size_t start = child.program.subtree(end);  
            string portion="";
            for (int j=start; j<=end; ++j)
            {
                portion += child.program.at(j)->name + " ";
            }
            logger.log("\t\tdelete mutating [ " + 
                    portion + " ] from program " +
                    child.program_str(), 3);

            NodeVector terms;  // potential fns 
            
            // find terminals with matching output types to i
            for (const auto& t: params.terminals)
            { 
                // find terms with matching output types that take 
                // this node type as arg
                if ( t->otype==child.program.at(i)->otype )
                { 
                    terms.push_back(t->rnd_clone());              
                }
            }

            if (terms.size()==0)  // if no insertion terminals match, skip
            {
                logger.log("\t\tnevermind, couldn't find a matching terminal",
                        3);
                continue;
            }

            // choose a function to insert                    
            std::unique_ptr<Node> insertion = random_node(terms);
            
            string s; 
            logger.log("\t\tinsertion: " + insertion->name + "\n", 3);

            // delete portion of program
            if (logger.get_log_level() >=3)
            { 
                std::string s="";
                for (unsigned i = start; i<=end; ++i) 
                {
                    s+= child.program.at(i)->name + " ";
                }
                logger.log("\t\tdeleting " + std::to_string(start) + " to " + 
                        std::to_string(end) + ": " + s, 3);
            }    
            child.program.erase(child.program.begin()+start,
                    child.program.begin()+end+1);

            // insert the terminal that was chosen 
            child.program.insert(child.program.begin()+start, 
                    insertion->clone());
            logger.log("\t\tresult of delete mutation: " + 
                    child.program_str(), 3);
            continue;
        }
        /* std::cout << "i: " << i << "\n"; */ 
    }
}

void Variation::delete_dimension_mutate(Individual& child, 
        const Parameters& params)
{
    /*! deletion mutation. works by pruning a dimension. 
     * @param child: individual to be mutated
     * @param params: parameters  
     * @return mutated child
     * */
    logger.log("\t\tprogram: " + child.program_str(),3);
    vector<size_t> roots = child.program.roots();
    
    size_t end = r.random_choice(roots,child.p); 
    size_t start = child.program.subtree(end);  
    if (logger.get_log_level() >= 3)
    { 
        std::string s="";
        for (unsigned i = start; i<=end; ++i) 
        {
            s+= child.program.at(i)->name + " ";
        }
        logger.log("\t\tdeleting " + std::to_string(start) + " to " + 
                std::to_string(end) + ": " + s, 3);
    }    
    child.program.erase(child.program.begin()+start,
            child.program.begin()+end+1);
    logger.log("\t\tresult of delete mutation: " + child.program_str(), 3);
}

bool Variation::correlation_delete_mutate(Individual& child, 
        MatrixXf Phi, const Parameters& params, const Data& d)
{
    /*! deletion mutation. works by pruning a dimension. 
     * the dimension that is pruned matches these criteria:
     * 1) it is in the pair of features most highly correlated. 
     * 2) it is less correlated with the dependent variable than its pair.
     * @param child: individual to be mutated
     * @param Phi: the behavior of the parent
     * @param params: parameters  
     * @param d: data
     * @return mutated child
     * */
    logger.log("\t\tprogram: " + child.program_str(),3); 
    // mean center features
    for (int i = 0; i < Phi.rows(); ++i)
    {
        Phi.row(i) = Phi.row(i).array() - Phi.row(i).mean();
    }
    /* cout << "Phi: " << Phi.rows() << "x" << Phi.cols() << "\n"; */
    // calculate highest pairwise correlation and store feature indices
    float highest_corr = 0;
    int f1=0, f2 = 0;
    for (int i = 0; i < Phi.rows()-1; ++i)
    {
        for (int j = i+1; j < Phi.rows(); ++j)
        {
           float corr = pearson_correlation(Phi.row(i).array(),
                        Phi.row(j).array());

           /* cout << "correlation (" << i << "," << j << "): " */ 
           /*     << corr << "\n"; */

           if (corr > highest_corr)
           {
               highest_corr = corr;
               f1 = i;
               f2 = j;
           }
        }
    }
    logger.log("chosen pair: " + to_string(f1) +  ", " + to_string(f2)
            + "; corr = " + to_string(highest_corr), 3);
    if (f1 == 0 && f2 == 0)
    {
        WARN("WARNING: couldn't get proper "
                "correlations. aborting correlation_delete_mutate\n");
        return false;
    }
    // pick the feature, f1 or f2, that is less correlated with y
    float corr_f1 = pearson_correlation(d.y.array()-d.y.mean(),
                                        Phi.row(f1).array()); 
    float corr_f2 = pearson_correlation(d.y.array()-d.y.mean(),
                                        Phi.row(f2).array()); 
    logger.log( "corr (" + to_string(f1) + ", y): " + to_string(corr_f1), 3);
    logger.log( "corr (" + to_string(f2) + ", y): " + to_string(corr_f2), 3);
    int choice = corr_f1 <= corr_f2 ? f1 : f2; 
    logger.log( "chose (" + to_string(choice), 3);
    // pick the subtree starting at roots(choice) and delete it
    vector<size_t> roots = child.program.roots();
    size_t end = roots.at(choice); 
    size_t start = child.program.subtree(end);  
    if (logger.get_log_level() >=3)
    { 
        std::string s="";
        for (unsigned i = start; i<=end; ++i) 
            s+= child.program.at(i)->name + " ";
        logger.log("\t\tdeleting " + std::to_string(start) + " to " + 
                std::to_string(end) + ": " + s, 3);
    }    
    child.program.erase(child.program.begin()+start,
            child.program.begin()+end+1);

    logger.log("\t\tresult of corr delete mutation: " 
               + child.program_str(), 3);

    if (!child.program.is_valid_program(d.X.rows(), 
                params.longitudinalMap))
    {
        cout << "Error in correlation_delete_mutate: child is not a valid "
             << "program.\n";
        cout << child.program_str() << endl;
        cout << child.get_eqn() << endl;
    }

    return highest_corr > 0.999;
}

bool Variation::cross(const Individual& mom, const Individual& dad, 
        Individual& child, const Parameters& params, const Data& d)
{
    /*!
     * crossover by either subtree crossover or swapping of dimensions. 
     *
     * @param   mom: root parent
     * @param   dad: parent from which subtree is chosen
     * @param   child: result of cross
     * @param   params: parameters
     *
     * @return  child: mom with dad subtree graft
     */
                
    // some of the time, do subtree xo. the reset of the time, 
    // do some version of root crossover.
    bool subtree = r() > params.root_xo_rate;                 
    vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
    size_t i1, j1, i2, j2;       // i1-j1: mom portion, i2-j2: dad portion
    
    if (subtree) 
    {
        logger.log("\tsubtree xo",3);
        // limit xo choices to matching output types in the programs. 
        vector<char> d_otypes;
        for (const auto& p : dad.program)
            if(!in(d_otypes,p->otype))
                d_otypes.push_back(p->otype);
        
        // get valid subtree locations
        for (size_t i =0; i<mom.size(); ++i) 
            if (in(d_otypes,mom.program.at(i)->otype)) 
                mlocs.push_back(i);  
        // mom and dad have no overlapping types, can't cross
        if (mlocs.size()==0)                
        {
            logger.log("WARNING: no overlapping types between " + 
                    mom.program_str() + "," + dad.program_str() + "\n", 3);
            return 0;               
        }
        j1 = r.random_choice(mlocs,mom.get_p(mlocs,true));    

        // get locations in dad's program that match the subtree type picked 
        // from mom
        for (size_t i =0; i<dad.size(); ++i) 
        {
            if (dad.program.at(i)->otype == mom.program.at(j1)->otype) 
                dlocs.push_back(i);
        }
    } 
    else             // half the time, pick a root node
    {
        if (params.residual_xo)
            return residual_cross(mom, dad, child, params, d);
        else if (params.stagewise_xo)
            return stagewise_cross(mom, dad, child, params, d);
        else
        {
            logger.log("\troot xo",3);
            mlocs = mom.program.roots();
            dlocs = dad.program.roots();
            logger.log("\t\trandom choice mlocs (size "+
                       std::to_string(mlocs.size())+"), p size: "+
                       std::to_string(mom.p.size()),3);
            // weighted probability choice    
            j1 = r.random_choice(mlocs,mom.get_p(mlocs));           
        }
    }
    /* cout << "mom subtree\t" << mom.program_str() << " starting at " */ 
    /*     << j1 << "\n"; */
    // get subtree              
    i1 = mom.program.subtree(j1);
    /* cout << "mom i1: " << i1 << endl; */                    
    /* cout << "dad subtree\n" << dad.program_str() << "\n"; */
    /* cout << "dad subtree\n"; */
    // get dad subtree
    j2 = r.random_choice(dlocs);
    i2 = dad.program.subtree(j2); 
           
    /* cout << "splice programs\n"; */
    // make child program by splicing mom and dad
    splice_programs(child.program, mom.program, i1, j1, dad.program, i2, j2 );
                 
    if (logger.get_log_level() >= 3)
        print_cross(mom,i1,j1,dad,i2,j2,child);     

    assert(child.program.is_valid_program(params.num_features, 
                params.longitudinalMap));
    // check child depth and dimensionality
    return (child.size() > 0 && child.size() <= params.max_size 
                && child.get_dim() <= params.max_dim);
}

/// residual crossover
bool Variation::residual_cross(const Individual& mom, const Individual& dad, 
        Individual& child, const Parameters& params, const Data& d)
{
    /*!
     * crossover by swapping in a dimension most correlated with the residual 
     * of mom. 
     *
     * @param   mom: root parent
     * @param   dad: parent from which subtree is chosen
     * @param   child: result of cross
     * @param   params: parameters
     *
     * @return  child: mom with dad subtree graft
     */
               
    logger.log("\tresidual xo",3);
    vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
    // i1-j1: mom portion, i2-j2: dad portion
    size_t i1, j1, j1_idx, i2, j2;       
    
    mlocs = mom.program.roots();
    vector<int> mlocs_indices(mlocs.size());
    std::iota(mlocs_indices.begin(),mlocs_indices.end(),0);

    dlocs = dad.program.roots();
    logger.log("\t\trandom choice mlocs (size "+
               std::to_string(mlocs.size())+"), p size: "+
               std::to_string(mom.p.size()),3);
    // weighted probability choice
    j1_idx = r.random_choice(mlocs_indices,mom.get_p(mlocs));
    j1 = mlocs.at(j1_idx); 
    // get subtree              
    i1 = mom.program.subtree(j1);
                        
    // get dad subtree
    // choose root in dad that is most correlated with the residual of
    // j2 = index in dad.Phi that maximizes R2(d.y, w*mom.Phi - w_i1*Phi_i1)
    /* cout << "mom: " << mom.get_eqn() << "\n"; */
    /* cout << "mom yhat: " << mom.yhat.transpose() << "\n"; */
    VectorXf tree = (mom.ml->get_weights().at(j1_idx) *
            mom.Phi.row(j1_idx).array());
    /* cout << "tree (idx=" << j1_idx << "): " << tree.transpose() << "\n"; */
    VectorXf mom_pred_minus_tree = mom.yhat - tree; 
    /* #pragma omp critical */
    /* { */
    /* VectorXf mom_pred_minus_tree = mom.predict_drop(d,params,j1_idx); */ 
    /* } */
    /* cout << "mom_pred_minus_tree: " << mom_pred_minus_tree.transpose() 
     * << "\n"; */
    VectorXf mom_residual = d.y - mom_pred_minus_tree;
    /* cout << "mom_residual: " << mom_residual.transpose() << "\n"; */
   
    // get correlations of dad's features with the residual from mom, 
    // less the swap choice
    vector<float> corrs(dad.Phi.rows());
    int best_corr_idx = 0;
    float best_corr = -1;
    float corr; 

    for (int i = 0; i < dad.Phi.rows(); ++i)
    {
        corr = pearson_correlation(mom_residual.array(), 
                dad.Phi.row(i).array());
        /* cout << "corr( " << i << "): " << corr << "\n"; */
        corrs.at(i) = corr; // this can be removed
        if (corr > best_corr )
        {
            best_corr_idx = i; 
            best_corr = corr;
        }
    }
    /* cout << "best_corr_idx: " << best_corr_idx << ", R^2: " 
     * << best_corr << "\n"; */
    /* cout << "corrs: "; */
    /* for (auto c : corrs) cout << c << ", "; cout << "\n"; */
    /* cout << "chose corr at " << best_corr_idx << "\n"; */
    j2 = dlocs.at(best_corr_idx);
    i2 = dad.program.subtree(j2); 
    
    if (logger.get_log_level() >= 3)
        print_cross(mom,i1,j1,dad,i2,j2,child,false);     
   
    // make child program by splicing mom and dad
    splice_programs(child.program, mom.program, i1, j1, dad.program, i2, j2 );
    
    if (logger.get_log_level() >= 3)
        print_cross(mom,i1,j1,dad,i2,j2,child,true);     
         
    
    assert(child.program.is_valid_program(params.num_features, 
                params.longitudinalMap));
    // check child depth and dimensionality
    return child.size()>0 && child.size() <= params.max_size 
                && child.get_dim() <= params.max_dim;
    
}

/// stagewise crossover
bool Variation::stagewise_cross(const Individual& mom, const Individual& dad, 
        Individual& child, const Parameters& params, const Data& d)
{
    /*!
     * crossover by forward stagewise selection, matching mom's dimensionality. 
     *
     * @param   mom: root parent
     * @param   dad: parent from which subtree is chosen
     * @param   child: result of cross
     * @param   params: parameters
     *
     * @return  child: mom with dad subtree graft
     * procedure: 
     * set the residual equal to target $y$. center means around zero for all 
     * $\phi$. 
     * set $\phi_A$ to be all subprograms in $\phi_{p0}$ and $\phi_{p1}$.
     * while $|\phi_{c}| < |\phi_{p0}|$: 
     *     - pick $\phi^*$ from $\phi_{A}$ which is most correlated with 
     *     $\mathbf{r}$. 
     *     - compute the least squares coefficient $b$ for $\phi^*$ fit to 
     *     $\mathbf{r}$. 
     *     - update $\mathbf{r} = r - b\phi^*$ 
     * $\phi_c$ = all $\phi^*$ that were chosen 
     */
    logger.log("\tstagewise xo",3);
    // normalize the residual 
    VectorXf R = d.y.array() - d.y.mean();
    /* cout << "R: " << R.norm() << "\n"; */
    if (mom.Phi.cols() != dad.Phi.cols())
    {
        cout << "!!WARNING!! mom.Phi.cols() = " << mom.Phi.cols() 
             << " and dad.Phi.cols() = " << dad.Phi.cols() << "\n";
        cout << " d.y size: " << d.y.size() << "\n";
        cout << "mom: " << mom.program_str() << "\n";
        cout << "dad: " << dad.program_str() << "\n";
        return false;
    }
    MatrixXf PhiA(mom.Phi.rows()+dad.Phi.rows(), mom.Phi.cols()); 
    PhiA << mom.Phi, 
            dad.Phi; 
    /* cout << "mom Phi: " << mom.Phi.rows() << "x" << mom.Phi.cols() << "\n"; */
    /* cout << "dad Phi: " << dad.Phi.rows() << "x" << dad.Phi.cols() << "\n"; */
    /* cout << "PhiA: " << PhiA.rows() << "x" << PhiA.cols() << "\n"; */ 
    // normalize Phi
    for (int i = 0; i < PhiA.rows(); ++i)
    {
        PhiA.row(i) = PhiA.row(i).array() - PhiA.row(i).mean();
        /*cout << "PhiA( " << i << ").mean(): " << PhiA.row(i).mean() << "\n";*/
    }
    vector<int> sel_idx;
    float best_corr_idx;
    unsigned nsel = 0;
    float deltaR = 1; // keep track of changes to the residual
    // only keep going when residual is reduced by at least tol
    float tol = 0.01;     
    /* cout << "R norm\t\tdeltaR\n"; */

    bool condition = true;
    while (condition)
    {
        float best_corr = -1;
        float corr;
        // correlation of PhiA with residual
        // pick most correlated (store index)
        for (int i = 0; i < PhiA.rows(); ++i)
        {
            if (!in(sel_idx,i))
            {
                corr = pearson_correlation(R.array(), PhiA.row(i).array());
                /* cout << "corr( " << i << "): " << corr << "\n"; */
                if (corr > best_corr)
                {
                    best_corr_idx = i; 
                    best_corr = corr;
                }
            }
        }
        if (best_corr > 0)
        {
            /* cout << "best_corr_idx: " << best_corr_idx << ", R^2: " 
             * << best_corr << "\n"; */
            // least squares coefficient of phi* w.r.t. R
            float b =  (covariance(PhiA.row(best_corr_idx),R) / 
                        variance(PhiA.row(best_corr_idx)));
            /* cout << "b: " << b << "\n"; */
            /* cout << "b*phi: " << b*PhiA.row(best_corr_idx) << "\n"; */
            deltaR = R.norm();
            R = R - b*PhiA.row(best_corr_idx).transpose();
            deltaR = (deltaR - R.norm()) / deltaR; 
            /* cout << "R: " << R.transpose() << "\n"; */
            /* cout << R.norm() << "\t\t" << deltaR << "\n"; */
            // select best correlation index
            if (!params.stagewise_xo_tol || deltaR >= tol)
            {
                sel_idx.push_back(best_corr_idx);
            }
        }
        ++nsel;
        /* if (deltaR < tol) */
        /* { */
        /*     cout << "!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!" */
        /*         << "\nHAH! I caught you, fiend!\n" */
        /*         << deltaR << " < " << tol << "\n"; */

        /* } */
        if (params.stagewise_xo_tol)
        {
            condition = (deltaR > tol 
                    && nsel <= (mom.Phi.rows() + dad.Phi.rows()));
        }
        else
        {
            condition = nsel < mom.Phi.rows() ;
        }
        /* cout << "condition: " << condition << "\n"; */
    }

    // compose a child from each feature referenced in sel_idx. 
    vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
    
    mlocs = mom.program.roots();
    dlocs = dad.program.roots();
    /* cout << "mlocs size: " << mlocs.size() << ", dlocs size: " 
     * << dlocs.size() << "\n"; */
    child.program.clear();

    for (int idx : sel_idx)
    {
        /* cout << "idx: " << idx << "\n"; */

        if (idx < mom.Phi.rows())
        {
            int stop = mlocs.at(idx);
            // get subtree indices
            int start = mom.program.subtree(stop);
            // construct child program
            /* cout << "adding mom.program (len= " << mom.program.size() 
             * << ") from " << start */
            /*     << " to " << stop << "\n"; */
            for (unsigned i = start; i <= stop ; ++i)        
            {
                child.program.push_back(mom.program.at(i)->clone());
            }
        }
        else
        {
            int stop = dlocs.at(idx - mom.Phi.rows());
            // get subtree indices
            int start = dad.program.subtree(stop);
            // construct child program
            /* cout << "adding dad.program (len= " << dad.program.size() 
             * << ") from " << start */
            /*     << " to " << stop << "\n"; */
            for (unsigned i = start; i <= stop ; ++i)        
            {
                child.program.push_back(dad.program.at(i)->clone());
            }
        }
        
    }
            
    /* cout << "child program size: " << child.program.size() << "\n"; */
    /* if (logger.get_log_level() >= 3) */
    /*     print_cross(mom,i1,j1,dad,i2,j2,child,false); */     
   
    /* // make child program by splicing mom and dad */
    /* splice_programs(child.program, mom.program, i1, j1, dad.program, 
     * i2, j2 ); */
    
    /* if (logger.get_log_level() >= 3) */
    /*     print_cross(mom,i1,j1,dad,i2,j2,child,true); */     

    /* cout << "asserting validity\n"; */
    assert(child.program.is_valid_program(params.num_features, 
                params.longitudinalMap));
    /* cout << "returning \n"; */
    // check child depth and dimensionality
    return child.size()>0 && child.size() <= params.max_size 
                && child.get_dim() <= params.max_dim;
    
}
// swap vector subsets with different sizes. 
void Variation::splice_programs( NodeVector& vnew,
                                 const NodeVector& v1, size_t i1, size_t j1, 
                                 const NodeVector& v2, size_t i2, size_t j2)
{
    /*!
     * swap vector subsets with different sizes. 
     * constructs a vector made of v1[0:i1], v2[i2:j2], v1[i1:end].
     *
     * @param   v1: root parent 
     * @param       i1: start of splicing segment 
     * @param       j1: end of splicing segment
     * @param   v2: donating parent
     * @param       i2: start of donation
     * @param       j2: end of donation
     *
     * @return  vnew: new vector 
     */
    logger.log("splice_programs",3);
    if (i1 >= v1.size())
        cout << "i1 ( " << i1 << ") >= v1 size (" << v1.size() << ")\n";
    if (i2 >= v2.size())
        cout << "i2 ( " << i2 << ") >= v2 size (" << v2.size() << ")\n";
    if (j1+1 < 0)
        cout << "j1+1 < 0 (j1 = " << j1 << endl;
    if (j2+1 < 0)
        cout << "j2+1 < 0 (j2 = " << j2 << endl;
    // size difference between subtrees  
    // beginning of v1
    try
    {
        for (unsigned i = 0; i < i1 ; ++i)                  
            vnew.push_back(v1.at(i)->clone());
        // spliced in v2 portion
        for (unsigned i = i2; i <= j2 ; ++i)                         
            vnew.push_back(v2.at(i)->clone());
        // end of v1
        for (unsigned i = j1+1; i < v1.size() ; ++i)                
            vnew.push_back(v1.at(i)->clone());
    }
    catch (std::bad_alloc& ba)
    {
        std::cerr << "bad_alloc caught: " << ba.what() << "\n";
    }
}

void Variation::print_cross(const Individual& mom, size_t i1, size_t j1, 
        const Individual& dad, size_t i2, size_t j2, Individual& child, bool after)
{
    std::cout << "\t\tattempting the following crossover:\n\t\t";
    for (int i =0; i<mom.program.size(); ++i){
       if (i== i1) 
           std::cout << "[";
       std::cout << mom.program.at(i)->name << " ";
       if (i==j1)
           std::cout <<"]";
    }
    std::cout << "\n\t\t";
   
    for (int i =0; i<dad.program.size(); ++i){
        if (i== i2) 
            std::cout << "[";
        std::cout << dad.program.at(i)->name << " ";
        if (i==j2)
            std::cout <<"]";
    }
    std::cout << "\n\t\t";
    if (after)
    {
        std::cout << "child after cross: ";
        for (unsigned i = 0; i< child.program.size(); ++i){
            if (i==i1) std::cout << "[";
            std::cout << child.program.at(i)->name << " ";
            if (i==i1+j2-i2) std::cout << "]";
        }
        std::cout << "\n";
    }
}

}
}
