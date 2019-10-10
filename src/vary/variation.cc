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
                // get random mom and dad 
                Individual& mom = pop.individuals.at(r.random_choice(parents));
                Individual& dad = pop.individuals.at(r.random_choice(parents));
                /* int dad = r.random_choice(parents); */
                // create child
                logger.log("\n===\ncrossing " + mom.get_eqn() + "\nwith\n " + 
                           dad.get_eqn() , 3);
               
                // perform crossover
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
                logger.log("mutating " + mom.get_eqn() + "(" + 
                        mom.program_str() + ")", 3);
                // create child
                pass = mutate(mom,child,params);
                
                logger.log("mutating " + mom.get_eqn() + " produced " + 
                        child.get_eqn() + ", pass: " + std::to_string(pass),3);
                child.set_parents({mom});
            }
            if (pass)
            {
                assert(child.size()>0);
                assert(pop.open_loc.size()>i-start);
                logger.log("assigning " + child.program_str() + 
                        " to pop.individuals[" + std::to_string(i) + 
                        "] with pop.open_loc[" + std::to_string(i-start) + 
                    "]=" + std::to_string(pop.open_loc[i-start]),3);

                pop.individuals[i] = child;
                pop.individuals[i].loc = pop.open_loc[i-start];                   
            }
        }    
   }
  
   pop.update_open_loc();
}

bool Variation::mutate(Individual& mom, Individual& child, 
        const Parameters& params)
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
    if (rf < 1.0/3.0 && child.get_dim() > 1){
        delete_mutate(child,params); 
        assert(child.program.is_valid_program(params.num_features, 
                    params.longitudinalMap));
    }
    else if (rf < 2.0/3.0 && child.size() < params.max_size)
    {
        insert_mutate(child,params);
        assert(child.program.is_valid_program(params.num_features, 
                    params.longitudinalMap));
    }
    else
    {        
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
    logger.log("\tpoint mutation",3);
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
                        f->arity['f']==p->arity['f'] && 
                        f->arity['b']==p->arity['b'] &&
                        f->arity['c']==p->arity['c'] &&
                        f->arity['z']==p->arity['z'])
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
                HANDLE_ERROR_NO_THROW("WARNING: couldn't mutate " + 
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
    
    logger.log("\tinsert mutation",3);
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
                        child.program[i]->name + " with probability " +
                        std::to_string(child.get_p(i)) + "/" + 
                        std::to_string(n), 3);
                NodeVector insertion;  // inserted segment
                NodeVector fns;  // potential fns 
                
                // find instructions with matching output types and a 
                // matching arity to i
                for (const auto& f: params.functions)
                { 
                    // find fns with matching output types that take 
                    // this node type as arg
                    if (f->arity[child.program[i]->otype] > 0 && 
                            f->otype==child.program[i]->otype )
                    { 
                        // make sure there are satisfactory types in n
                        // terminals to fill fns' args
                        if (child.program[i]->otype=='b')
                            if (in(params.dtypes,'b') || f->arity['b']==1)
                                fns.push_back(f->rnd_clone());
                        else if (child.program[i]->otype=='f')
                            if (f->arity['b']==0 || in(params.dtypes,'b') )
                                fns.push_back(f->rnd_clone());              
                    }
                }

                if (fns.size()==0)  // if no insertion functions match, skip
                    continue;

                // choose a function to insert                    
                insertion.push_back(random_node(fns));
                
                unsigned fa = insertion.back()->arity['f']; // float arity
                unsigned ca = insertion.back()->arity['c']; //categorical arity
                unsigned ba = insertion.back()->arity['b']; // bool arity
                // decrement function arity by one for child node 
                if (child.program[i]->otype=='f') --fa;
                else if (child.program[i]->otype=='c') --ca;
                else --ba; 
                
                // push back new arguments for the rest of the function
                for (unsigned j = 0; j< fa; ++j)
                    insertion.make_tree(params.functions,params.terminals,0,
                              params.term_weights,params.op_weights,'f',
                              params.ttypes);
                // add the child now if float
                if (child.program[i]->otype=='f')                        
                    insertion.push_back(child.program[i]->clone());
                for (unsigned j = 0; j< ca; ++j)
                    insertion.make_tree(params.functions,params.terminals,0,
                              params.term_weights,params.op_weights,'c',
                              params.ttypes);
                // add the child now if categorical
                if (child.program[i]->otype=='c')                        
                    insertion.push_back(child.program[i]->clone());
                for (unsigned j = 0; j< ba; ++j)
                    insertion.make_tree(params.functions,params.terminals,0,
                              params.term_weights,params.op_weights,'b',
                              params.ttypes);
                if (child.program[i]->otype=='b') // add the child now if bool
                    insertion.push_back(child.program[i]->clone());
                // post-fix notation
                std::reverse(insertion.begin(),insertion.end());
               
                string s; 
                for (const auto& ins : insertion) s += ins->name + " "; 
                logger.log("\t\tinsertion: " + s + "\n", 3);
                NodeVector new_program; 
                splice_programs(new_program, 
                                child.program, i, i, 
                                insertion, size_t(0), insertion.size()-1);
                child.program=new_program;
                i += insertion.size()-1;
            }
            /* std::cout << "i: " << i << "\n"; */ 
        }
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

void Variation::delete_mutate(Individual& child, const Parameters& params)
{
    /*! deletion mutation. works by pruning a dimension. 
     * @param child: individual to be mutated
     * @param params: parameters  
     * @return mutated child
     * */
    logger.log("\tdeletion mutation",3);
    logger.log("\t\tprogram: " + child.program_str(),3);
    vector<size_t> roots = child.program.roots();
    
    size_t end = r.random_choice(roots,child.p); 
    size_t start = child.program.subtree(end);  
    if (logger.get_log_level() >=3)
    { 
        std::string s="";
        for (unsigned i = start; i<=end; ++i) s+= child.program[i]->name + " ";
        logger.log("\t\tdeleting " + std::to_string(start) + " to " + 
                std::to_string(end) + ": " + s, 3);
    }    
    child.program.erase(child.program.begin()+start,
            child.program.begin()+end+1);
    logger.log("\t\tresult of delete mutation: " + child.program_str(), 3);
}

void Variation::correlation_delete_mutate(Individual& child, 
        const Parameters& params, const Data& d)
{
    /*! deletion mutation. works by pruning a dimension. 
     * the dimension that is pruned matches these criteria:
     * 1) it is in the pair of features most highly correlated. 
     * 2) it is less correlated with the dependent variable than its pair.
     * @param child: individual to be mutated
     * @param params: parameters  
     * @param d: data
     * @return mutated child
     * */
    logger.log("\tdeletion mutation",3);
    logger.log("\t\tprogram: " + child.program_str(),3);
    

    MatrixXf Phi = child.Phi;
    // mean center features
    for (int i = 0; i < Phi.rows(); ++i)
    {
        Phi.row(i) = Phi.row(i).array() - Phi.row(i).mean();
    }
    // calculate highest pairwise correlation and store feature indices
    float highest_corr = 0;
    int f1, f2;
    for (int i = 0; i < Phi.rows()-1; ++i)
    {
        for (int j = i+1; j < Phi.rows(); ++j)
        {
           float corr = pearson_correlation(Phi.row(i).array(),
                        Phi.row(j).array());
           if (corr > highest_corr)
           {
               highest_corr = corr;
               f1 = i;
               f2 = j;
           }
        }
    }
    // pick the feature, f1 or f2, that is less correlated with y
    float corr_f1 = pearson_correlation(d.y.array()-d.y.mean(),
                                        Phi.row(f1).array()); 
    float corr_f2 = pearson_correlation(d.y.array()-d.y.mean(),
                                        Phi.row(f2).array()); 
    int choice = corr_f1 <= corr_f2 ? f1 : f2; 
    // pick the subtree starting at roots(choice) and delete it
    vector<size_t> roots = child.program.roots();
    size_t end = roots.at(choice); 
    size_t start = child.program.subtree(end);  
    if (logger.get_log_level() >=3)
    { 
        std::string s="";
        for (unsigned i = start; i<=end; ++i) s+= child.program[i]->name + " ";
        logger.log("\t\tdeleting " + std::to_string(start) + " to " + 
                std::to_string(end) + ": " + s, 3);
    }    
    child.program.erase(child.program.begin()+start,
            child.program.begin()+end+1);
    logger.log("\t\tresult of corr delete mutation: " + child.program_str(), 3);
}

bool Variation::cross(Individual& mom, Individual& dad, Individual& child, 
                      const Parameters& params, const Data& d)
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
            if (in(d_otypes,mom.program[i]->otype)) 
                mlocs.push_back(i);  
        // mom and dad have no overlapping types, can't cross
        if (mlocs.size()==0)                {
            logger.log("WARNING: no overlapping types between " + 
                    mom.program_str() + "," + dad.program_str() + "\n", 3);
            return 0;               
        }
        j1 = r.random_choice(mlocs,mom.get_p(mlocs,true));    

        // get locations in dad's program that match the subtree type picked 
        // from mom
        for (size_t i =0; i<dad.size(); ++i) 
        {
            if (dad.program[i]->otype == mom.program[j1]->otype) 
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
            j1 = r.random_choice(mlocs,mom.get_p(mlocs));           }
    }
    /* cout << "mom subtree\t" << mom.program_str() << "\n"; */
    // get subtree              
    i1 = mom.program.subtree(j1);
                        
    /* cout << "dad subtree\n" << dad.program_str() << "\n"; */
    // get dad subtree
    j2 = r.random_choice(dlocs);
    i2 = dad.program.subtree(j2); 
           
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
bool Variation::residual_cross(Individual& mom, Individual& dad, 
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
               
    vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
    // i1-j1: mom portion, i2-j2: dad portion
    size_t i1, j1, j1_idx, i2, j2;       
    
    logger.log("\tresidual xo",3);
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
bool Variation::stagewise_cross(Individual& mom, Individual& dad, 
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
    /* cout << "R: " << R.transpose() << "\n"; */
    /* cout << "R mean: " << R.mean() << "\n"; */
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
    /*cout << "mom Phi: " << mom.Phi.rows() << "x" << mom.Phi.cols() << "\n";*/
    /*cout << "dad Phi: " << dad.Phi.rows() << "x" << dad.Phi.cols() << "\n";*/
    /*cout << "PhiA: " << PhiA.rows() << "x" << PhiA.cols() << "\n"; */
    // normalize Phi
    for (int i = 0; i < PhiA.rows(); ++i)
    {
        PhiA.row(i) = PhiA.row(i).array() - PhiA.row(i).mean();
        /*cout << "PhiA( " << i << ").mean(): " << PhiA.row(i).mean() << "\n";*/
    }
    vector<int> sel_idx;
    float best_corr_idx;
    unsigned nsel = 0;
    while (nsel < mom.Phi.rows())
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
            R = R - b*PhiA.row(best_corr_idx).transpose();
            /* cout << "R: " << R.transpose() << "\n"; */
            /* cout << "R norm: " << R.norm() << "\n"; */
            // select best correlation index
            sel_idx.push_back(best_corr_idx);
        }
        ++nsel;
    }
    /* cout << "sel_idx: "; */
    /* for (auto s: sel_idx) cout << s << ","; cout << "\n"; */
    // take stored indices and find corresponding program positions for them
    //
    /* std::sort(sel_idx.begin(), sel_idx.end()); */
    // TODO: compose a child from each feature referenced in sel_idx. 
    // 
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
    /* std::cout << "in splice_programs\n"; */
    /* std::cout << "i1: " << i1 << ", j1: " << j1  << ", i2: " << i2 
     * << ", j2:" << j2 << "\n"; */
    // size difference between subtrees  
    // beginning of v1
    for (unsigned i = 0; i < i1 ; ++i)                  
        vnew.push_back(v1.at(i)->clone());
    // spliced in v2 portion
    for (unsigned i = i2; i <= j2 ; ++i)                         
        vnew.push_back(v2.at(i)->clone());
    // end of v1
    for (unsigned i = j1+1; i < v1.size() ; ++i)                
        vnew.push_back(v1.at(i)->clone());
}

void Variation::print_cross(Individual& mom, size_t i1, size_t j1, 
        Individual& dad, size_t i2, size_t j2, Individual& child, bool after)
{
    std::cout << "\t\tattempting the following crossover:\n\t\t";
    for (int i =0; i<mom.program.size(); ++i){
       if (i== i1) 
           std::cout << "[";
       std::cout << mom.program[i]->name << " ";
       if (i==j1)
           std::cout <<"]";
    }
    std::cout << "\n\t\t";
   
    for (int i =0; i<dad.program.size(); ++i){
        if (i== i2) 
            std::cout << "[";
        std::cout << dad.program[i]->name << " ";
        if (i==j2)
            std::cout <<"]";
    }
    std::cout << "\n\t\t";
    if (after)
    {
        std::cout << "child after cross: ";
        for (unsigned i = 0; i< child.program.size(); ++i){
            if (i==i1) std::cout << "[";
            std::cout << child.program[i]->name << " ";
            if (i==i1+j2-i2) std::cout << "]";
        }
        std::cout << "\n";
    }
}

}}
