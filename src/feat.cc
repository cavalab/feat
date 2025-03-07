/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "feat.h"

//shogun initialization
void __attribute__ ((constructor)) ctor()
{
    init_shogun_with_defaults();
}

void __attribute__ ((destructor))  dtor()
{
    exit_shogun();
    FT::Rnd::destroy();
    FT::Logger::destroy();
}

using namespace FT;
    
/// @brief initialize Feat object for fitting.
void Feat::init()
{
    if (params.n_jobs!=0)
        omp_set_num_threads(params.n_jobs);
    r.set_seed(params.random_state);

    if (GPU)
        initialize_cuda();
    // set Feat's Normalizer to only normalize floats by default
    this->N = Normalizer(false);
    this->archive.set_objectives(params.objectives);
    set_is_fitted(false);

    // start the clock
    timer.Reset();
    // signal handler
    signal(SIGINT, my_handler);
    // reset statistics
    this->stats = Log_Stats();
    params.use_batch = params.bp.batch_size>0;
}

void Feat::fit(MatrixXf& X, VectorXf& y, LongData& Z)
{

    /*! 
     *  Input:
     
     *       X: n_features x n_samples MatrixXf of features
     *       y: VectorXf of labels 
     
     *  Output:
     
     *       updates best_estimator, hof
    
     *   steps:
     *	   1. fit model yhat = f(X)
     *	   2. generate transformations Phi(X) for each individual
     *	   3. fit model yhat_new = f( Phi(X)) for each individual
     *	   4. evaluate features
     *	   5. selection parents
     *	   6. produce offspring from parents via variation
     *	   7. select surviving individuals from parents and offspring
     */
    this->init();
    std::ofstream log;                      ///< log file stream
    if (!logfile.empty())
        log.open(logfile, std::ofstream::app);
    params.init(X, y);       

    string FEAT;
    if (params.verbosity == 1)
    {
        FEAT = (  
      "/// Feature Engineering Automation Tool " 
      "* \xc2\xa9 La Cava et al 2017 "
      "* GPL3 \\\\\\\n"
        );
    }
    else if (params.verbosity == 2)
    {
        FEAT = (  
      "/////////////////////////////////////////////////////////////////////\n"
      "//           * Feature Engineering Automation Tool *               //\n"
      "// La Cava et al. 2017                                             //\n"
      "// License: GPL v3                                                 //\n"
      "// https://cavalab.org/feat                                        //\n"
      "/////////////////////////////////////////////////////////////////////\n"
        );
    }

    if (params.use_batch)
    {
        if (params.bp.batch_size >= X.cols())
        {
            logger.log("turning off batch because X has fewer than " 
                    + to_string(params.bp.batch_size) + " samples", 1);
            params.use_batch = false;
        }
        else
        {
            logger.log("using batch with batch_size= " 
                    + to_string(params.bp.batch_size), 2);
        }
    }
    
    // if(str_dim.compare("") != 0)
    // {
    //     string dimension;
    //     dimension = str_dim.substr(0, str_dim.length() - 1);
    //     logger.log("STR DIM IS "+ dimension, 2);
    //     logger.log("Cols are " + std::to_string(X.rows()), 2);
    //     logger.log("Setting dimensionality as " + 
    //                std::to_string((int)(ceil(stod(dimension)*X.rows()))), 2);
    //     set_max_dim(ceil(stod(dimension)*X.rows()));
    // }
    

    logger.log(FEAT,1);
    
    this->archive.set_objectives(params.objectives);

    // normalize data
    if (params.normalize)
    {
        N.fit_normalize(X,params.dtypes);                   
    }
    this->pop = Population(params.pop_size);
    this->evaluator = Evaluation(params.scorer_);

    /* create an archive to save Pareto front, 
     * unless NSGA-2 is being used for survival 
     */
    /* if (!survival.compare("nsga2")) */
    /*     use_arch = false; */
    /* else */
    /*     use_arch = true; */
    use_arch = false;

    logger.log("scorer: " + params.scorer_, 1);

    // split data into training and test sets
    //Data data(X, y, Z, params.classification);
    DataRef d(X, y, Z, params.classification, params.protected_groups);
    //DataRef d;
    //d.setOriginalData(&data);
    d.train_test_split(params.shuffle, params.split);
    // define terminals based on size of X
    params.set_terminals(d.o->X.rows(), d.o->Z);        

    // initial model on raw input
    logger.log("Setting up data", 2);
    float t0 =  timer.Elapsed().count();
    
    //data for batch training
    MatrixXf Xb;
    VectorXf yb;
    LongData Zb;
    Data db(Xb, yb, Zb, params.classification, params.protected_groups);
    
    Data *tmp_train;
    
    if(params.use_batch)
    {
        tmp_train = d.t;
        d.t->get_batch(db, params.bp.batch_size);
        d.setTrainingData(&db);
    }
    
    if (params.classification) 
        params.set_sample_weights(d.t->y); 
    

    // initialize population 
    ////////////////////////
    logger.log("Initializing population", 2);
   
    bool random = selector.get_type() == "random";

    // initial model
    ////////////////
    logger.log("Fitting initial model", 2);
    t0 =  timer.Elapsed().count();
    initial_model(d);  
    logger.log("Initial fitting took " 
            + std::to_string(timer.Elapsed().count() - t0) + " seconds",2);

    // initialize population with initial model and/or starting pop
    pop.init(best_ind,params,random, this->starting_pop);
    logger.log("Initial population:\n"+pop.print_eqns(),3);

    // evaluate initial population
    logger.log("Evaluating initial population",2);
    evaluator.fitness(pop.individuals,*d.t,params);
    evaluator.validation(pop.individuals,*d.v,params);
    
    logger.log("Initial population done",2);
    logger.log(std::to_string(timer.Elapsed().count()) + " seconds",2);
    
    vector<size_t> survivors;
    
    if(params.use_batch)    // reset d to all training data
        d.setTrainingData(tmp_train, true);

    // =====================
    // main generational loop
    unsigned g = 0;
    unsigned stall_count = 0;
    float fraction = 0;
    // continue until max gens is reached or max_time is up (if it is set)
    
    while(
        // time limit
        (params.max_time == -1 || params.max_time > timer.Elapsed().count())
        // generation limit
        && g<params.gens                                                    
        // stall limit
        && (params.max_stall == 0 || stall_count < params.max_stall) 
        )      
    {
        fraction = params.max_time == -1 ? ((g+1)*1.0)/params.gens : 
                                       timer.Elapsed().count()/params.max_time;
        if(params.use_batch)
        {
            d.t->get_batch(db, params.bp.batch_size);
            DataRef dbr;    // reference to minibatch data
            dbr.setTrainingData(&db);
            dbr.setValidationData(d.v);

            if (params.classification)
                params.set_sample_weights(dbr.t->y); 

            run_generation(g, survivors, dbr, log, fraction, stall_count);
        }
        else
        {
            run_generation(g, survivors, d, log, fraction, stall_count);
        }
        
        g++;
    }
    // =====================
    if ( params.max_stall != 0 && stall_count >= params.max_stall)
        logger.log("learning stalled",2);
    else if ( g >= params.gens) 
        logger.log("generation limit reached",2);
    else
        logger.log("max time reached",2);

    logger.log("train score: " + std::to_string(this->min_loss), 2);
    logger.log("validation score: " + std::to_string(min_loss_v), 2);
    logger.log("fitting final model to all training data...",2);


    // simplify the final model
    if (simplify > 0.0)
    {
        this->best_ind.fit(*d.o, params);
        simplify_model(d, this->best_ind);
    }

    // fit final model to best features
    final_model(d);   

    // if we're not using an archive, let's store the final population in the 
    // archive
    if (!use_arch)
    {
        archive.individuals = pop.individuals;
    }

    if (save_pop > 0)
    {
        pop.save(this->logfile+".pop.gen" + to_string(params.current_gen) 
                + ".json");
        this->best_ind.save(this->logfile+".best.json");
    }
    
    if (log.is_open())
        log.close();

    set_is_fitted(true);
    logger.log("Run Completed. Total time taken is " 
            + std::to_string(timer.Elapsed().count()) + " seconds", 1);
    logger.log("best model: " + this->get_eqn(),1);
    logger.log("tabular model:\n" + this->get_model(),2);
    logger.log("/// ----------------------------------------------------------------- \\\\\\",
            1);

}
/// set size of population 
void Feat::set_pop_size(int pop_size){ params.pop_size = pop_size; }            

/// set size of max generations              
void Feat::set_gens(int gens){ params.gens = gens;}         
            
/// set ML algorithm to use              
void Feat::set_ml(string ml){ params.ml = ml; }            

/// set EProblemType for shogun              
void Feat::set_classification(bool classification)
{
    params.classification = classification;
}
     
/// set level of debug info              
void Feat::set_verbosity(int verbosity){ params.set_verbosity(verbosity); }
            
/// set maximum stall in learning, in generations
void Feat::set_max_stall(int max_stall){	params.max_stall = max_stall; }
            
/// set selection method              
void Feat::set_selection(string sel){ this->selector = Selection(sel, false); }
            
/// set survivability              
void Feat::set_survival(string surv)
{ 
    survival=surv; 
    survivor = Selection(surv, true);
}
            
/// set cross rate in variation              
void Feat::set_cross_rate(float cross_rate)
{
    params.cross_rate = cross_rate; 
    variator.set_cross_rate(cross_rate);
}
            
/// set root cross rate in variation              
void Feat::set_root_xo_rate(float cross_rate)
{
    params.root_xo_rate = cross_rate; 
}

/// set program output type ('f', 'b')              
void Feat::set_otype(char ot){ params.set_otype(ot); }
            
            
/// set max depth of programs              
void Feat::set_max_depth(unsigned int max_depth)
{ 
    params.set_max_depth(max_depth); 
}
 
/// set maximum dimensionality of programs              
void Feat::set_max_dim(unsigned int max_dim){	params.set_max_dim(max_dim); }

///set dimensionality as multiple of the number of columns
// void Feat::set_max_dim(string str){ str_dim = str; }            

/// set seeds for each core's random number generator              
void Feat::set_random_state(int rs)
{ 
    params.random_state=rs;
    r.set_seed(rs); 
}
            
/// flag to set whether to use variable or constants for terminals              
void Feat::set_erc(bool erc){ params.erc = erc; }

/// flag to shuffle the input samples for train/test splits
void Feat::set_shuffle(bool sh){params.shuffle = sh;}

/// set train fraction of dataset
void Feat::set_split(float sp){params.split = sp;}

///set data types for input parameters
void Feat::set_dtypes(vector<char> dtypes){params.dtypes = dtypes;}

///set feedback
void Feat::set_fb(float fb){ params.feedback = fb;}

///set name for files
void Feat::set_logfile(string s){logfile = s;}

///set scoring function
void Feat::set_scorer(string s){params.set_scorer(s);}
string Feat::get_scorer_(){return params.scorer_;}
string Feat::get_scorer(){return params.scorer;}

/// set constant optimization options
void Feat::set_backprop(bool bp){params.backprop=bp;}

void Feat::set_simplify(float s){this->simplify=s;}

void Feat::set_corr_delete_mutate(bool s){this->params.corr_delete_mutate=s;}

void Feat::set_hillclimb(bool hc){params.hillclimb=hc;}

void Feat::set_iters(int iters){params.bp.iters = iters; params.hc.iters=iters;}

void Feat::set_lr(float lr){params.bp.learning_rate = lr;}

void Feat::set_batch_size(int bs)
{
    params.bp.batch_size = bs; 
    params.use_batch = bs>0;
}
 
///set number of threads
void Feat::set_n_jobs(unsigned t){ omp_set_num_threads(t); }

void Feat::set_max_time(int time){ params.max_time = time; }

void Feat::set_use_batch(){ params.use_batch = true; }

void Feat::set_protected_groups(string pg)
{
    params.set_protected_groups(pg);
}
/*                                                      
 * getting functions
 */

///return population size
int Feat::get_pop_size(){ return params.pop_size; }

///return size of max generations
int Feat::get_gens(){ return params.gens; }

///return ML algorithm string
string Feat::get_ml(){ return params.ml; }

///return type of classification flag set
bool Feat::get_classification(){ return params.classification; }

///return maximum stall in learning, in generations
int Feat::get_max_stall() { return params.max_stall; }

///return program output type ('f', 'b')             
vector<char> Feat::get_otypes(){ return params.otypes; }

///return current verbosity level set
int Feat::get_verbosity(){ return params.verbosity; }

///return max_depth of programs
int Feat::get_max_depth(){ return params.max_depth; }

///return cross rate for variation
float Feat::get_cross_rate(){ return params.cross_rate; }

///return max size of programs
int Feat::get_max_size(){ return params.max_size; }

///return max dimensionality of programs
int Feat::get_max_dim(){ return params.max_dim; }

///return boolean value of erc flag
bool Feat::get_erc(){ return params.erc; }

/// get name
string Feat::get_logfile(){ return logfile; }

///return number of features
int Feat::get_num_features(){ return params.num_features; }

///return whether option to shuffle the data is set or not
bool Feat::get_shuffle(){ return params.shuffle; }

///return fraction of data to use for training
float Feat::get_split(){ return params.split; }

///add custom node into feat
/* void add_function(unique_ptr<Node> N){ params.functions.push_back(N->clone()); } */

///return data types for input parameters
vector<char> Feat::get_dtypes(){ return params.dtypes; }

///return feedback setting
float Feat::get_fb(){ return params.feedback; }

///return best model
string Feat::get_representation(){ return best_ind.get_eqn();}

string Feat::get_eqn(bool sort){ return this->get_ind_eqn(sort, this->best_ind); };

string Feat::get_ind_eqn(bool sort, Individual& ind) 
{   
    vector<string> features = ind.get_features();
    vector<float> weights = ind.ml->get_weights();
    float offset = ind.ml->get_bias(); 
    
    /* if (params.normalize) */
    /* { */
    /*     offset = this->N.adjust_offset(weights, offset); */
    /*     this->N.adjust_weights(weights); */
    /* } */

    vector<size_t> order(weights.size());
    if (sort)
    {
        vector<float> aweights(weights.size());
        for (int i =0; i<aweights.size(); ++i) 
            aweights[i] = fabs(weights[i]);
        order = argsort(aweights, false);
    }
    else
        iota(order.begin(), order.end(), 0);

    string output;
    output +=  to_string(offset);
    if (weights.size() > 0)
    {
        if (weights.at(order.at(0)) > 0)
            output += "+";
    }
    int i = 0;
    for (const auto& o : order)
    {
        output += to_string(weights.at(o), 2);
        output += "*";
        output += features.at(o);
        if (i < order.size()-1)
        {
            if (weights.at(order.at(i+1)) > 0)
                output+= "+";
        }
        ++i;
    }

    return output;
}

string Feat::get_model(bool sort)
{   
    vector<string> features = best_ind.get_features();
    vector<float> weights = best_ind.ml->get_weights();
    float offset = best_ind.ml->get_bias(); 
    /* if (params.normalize) */
    /* { */
    /*     offset = this->N.adjust_offset(weights, offset); */
    /*     this->N.adjust_weights(weights); */
    /* } */

    vector<size_t> order(weights.size());
    if (sort)
    {
        vector<float> aweights(weights.size());
        for (int i =0; i<aweights.size(); ++i) 
            aweights[i] = fabs(weights[i]);
        order = argsort(aweights, false);
    }
    else
        iota(order.begin(), order.end(), 0);

    string output;
    output += "Weight\tFeature\n";
    output +=  to_string(offset) + "\toffset" + "\n";
    for (const auto& o : order)
    {
        output += to_string(weights.at(o), 2);
        output += "\t";
        output += features.at(o);
        output += "\n";
    }

    return output;
}

///get number of parameters in best
int Feat::get_n_params(){ return best_ind.get_n_params(); } 

///get dimensionality of best
int Feat::get_dim(){ return best_ind.get_dim(); } 

///get dimensionality of best
int Feat::get_complexity(){
    // Making sure it is calculated before returning it
    if (best_ind.get_complexity()==0)
        best_ind.set_complexity();
    return best_ind.get_complexity();
} 

/// return the number of nodes in the best model
int Feat::get_n_nodes(){ return best_ind.program.size(); }

///return population as string
vector<json> Feat::get_archive(bool front)
{
    /* TODO: maybe this should just return the to_json call of
     * the underlying population / archive. I guess the problem
     * is that we don't have to_json defined for vector<Individual>.
     */
    vector<Individual>* printed_pop = NULL; 

    string r = "";

    vector<size_t> idx;
    bool subset = false;
    if (front)  // only return individuals on the Pareto front
    {
        if (use_arch)
        {
            printed_pop = &archive.individuals;
        }
        else
        {
            unsigned n = 1;
            subset = true;
            idx = this->pop.sorted_front(n);
            printed_pop = &this->pop.individuals;
        }
    }
    else
        printed_pop = &this->pop.individuals;

    if (!subset)
    {
        idx.resize(printed_pop->size());
        std::iota(idx.begin(), idx.end(), 0);
    }

    bool includes_best_ind = false;

    vector<json> json_archive;

    for (int i = 0; i < idx.size(); ++i)
    {
        Individual& ind = printed_pop->at(idx[i]); 

        json j;
        to_json(j, ind);

        // r += j.dump();
        json_archive.push_back(j);

        if (i < idx.size() -1)
            r += "\n";
        // check if best_ind is in here
        if (ind.id == best_ind.id)
            includes_best_ind = true;
    }

    // add best_ind, if it is not included
    if (!includes_best_ind) 
    {
        json j;
        to_json(j, best_ind);
        json_archive.push_back(j);
    }

    // delete pop pointer
    printed_pop = NULL;
    delete printed_pop;
    
    return json_archive;
}

/// return the coefficients or importance scores of the best model. 
ArrayXf Feat::get_coefs()
{
    auto tmpw = best_ind.ml->get_weights();
    ArrayXf w = ArrayXf::Map(tmpw.data(), tmpw.size());
    return w;
}

/// get longitudinal data from file s
std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf>>> Feat::get_Z(string s, 
        int * idx, int idx_size)
{
    LongData Z;
    vector<int> ids(idx,idx+idx_size);
    load_partial_longitudinal(s,Z,',',ids);
        
    return Z;
}

            
void Feat::fit(MatrixXf& X, VectorXf& y)
{
    auto Z = LongData();
    fit(X,y,Z);
}


void Feat::run_generation(unsigned int g,
                      vector<size_t> survivors,
                      DataRef &d,
                      std::ofstream &log,
                      float fraction,
                      unsigned& stall_count)
{
    d.t->set_protected_groups();

    params.set_current_gen(g);

    // select parents
    logger.log("selection..", 2);
    vector<size_t> parents = selector.select(pop, params, *d.t);
    logger.log("parents:\n"+pop.print_eqns(), 3);          
    
    // variation to produce offspring
    logger.log("variation...", 2);
    variator.vary(pop, parents, params,*d.t);
    logger.log("offspring:\n" + pop.print_eqns(true), 3);

    // evaluate offspring
    logger.log("evaluating offspring...", 2);
    evaluator.fitness(pop.individuals, *d.t, params, true);
    evaluator.validation(pop.individuals, *d.v, params, true);

    // select survivors from combined pool of parents and offspring
    logger.log("survival...", 2);
    survivors = survivor.survive(pop, params, *d.t);
   
    // reduce population to survivors
    logger.log("shrinking pop to survivors...",2);
    pop.update(survivors);
    logger.log("survivors:\n" + pop.print_eqns(), 3);
    
    // we need to update best, so min_loss_v is updated inside stats
    logger.log("update best...",2);
    bool updated_best = update_best(d);

    if (params.max_stall > 0)
        update_stall_count(stall_count, updated_best);

    logger.log("update objectives...",2);
    if ( (use_arch || params.verbosity>1) || !logfile.empty()) {
        // set objectives to make sure they are reported in log/verbose/arch
        #pragma omp parallel for
        for (unsigned int i=0; i<pop.size(); ++i)
            pop.individuals.at(i).set_obj(params.objectives);
    }

    logger.log("update archive...",2);
    if (use_arch) 
        archive.update(pop,params);
    
    logger.log("calculate stats...",2);
    calculate_stats(d);

    if(params.verbosity>1)
        print_stats(log, fraction);    
    else if(params.verbosity == 1)
        printProgress(fraction);
    
    if (!logfile.empty())
        log_stats(log);

    if (save_pop > 1)
        pop.save(this->logfile+".pop.gen" + 
                    to_string(params.current_gen) + ".json");

    // tighten learning rate for grad descent as evolution progresses
    if (params.backprop)
    {
        params.bp.learning_rate = \
            (1-1/(1+float(params.gens)))*params.bp.learning_rate;
        logger.log("learning rate: " 
                + std::to_string(params.bp.learning_rate),3);
    }
    logger.log("finished with generation...",2);

}

void Feat::update_stall_count(unsigned& stall_count, bool best_updated)
{
    if (params.current_gen == 0 || best_updated )
    {
        /* best_med_score = this->med_loss_v; */
        stall_count = 0;
    }
    else
    {
        ++stall_count;
    }

    logger.log("stall count: " + std::to_string(stall_count), 2);
}


void Feat::final_model(DataRef& d)	
{
    // fits final model to best tranformation found.
    shared_ptr<CLabels> yhat;
    if (params.tune_final)
        yhat = best_ind.fit_tune(*d.o, params);
    else
        yhat = best_ind.fit(*d.o, params);

    VectorXf tmp;
    /* params.set_sample_weights(y);   // need to set new sample weights for y, */ 
                                    // which is probably from a validation set
    float score = evaluator.S.score(d.o->y,yhat,tmp,params.class_weights);
    logger.log("final_model score: " + std::to_string(score),2);
}

void Feat::simplify_model(DataRef& d, Individual& ind)
{
    /* Simplifies the final model using some expert rules and stochastic hill 
     * climbing. 
     * Expert rules:
     *  - NOT(NOT(x)) simplifies to x
     * Stochastic hill climbing:
     * for some number iterations, apply delete mutation to the equation. 
     * if the output of the model doesn't change, keep the mutations.
     */

    //////////////////////////////
    // check for specific patterns
    //////////////////////////////
    //
    Individual tmp_ind = ind;
    int starting_size = ind.size();
    vector<size_t> roots = tmp_ind.program.roots();
    vector<size_t> idx_to_remove;

    logger.log("\n=========\ndoing pattern pruning...",2);
    logger.log("simplify: " + to_string(this->simplify), 2);

    for (auto r : roots)
    {
        size_t start = tmp_ind.program.subtree(r);
        int first_occurence = -2;

        /* cout << "start: " << start << "\n"; */
        for (int i = start ; i <= r; ++i)
        {
            /* cout << "i: " << i << ", first_occurence: " << first_occurence */ 
            /*     << "\n"; */
            if (tmp_ind.program.at(i)->name.compare("not")==0)
            {
                if (first_occurence == i-1) // indicates two NOTs in a row
                {
                    /* cout << "pushing back " << first_occurence */ 
                    /*     << " and " << i << " to idx_to_remove\n"; */
                    idx_to_remove.push_back(first_occurence);
                    idx_to_remove.push_back(i);
                    // reset first_occurence so we don't pick up triple nots
                    first_occurence = -2;
                }
                else
                {
                    first_occurence = i; 
                }
            }
        }
    }
    // remove indices in reverse order so they don't change
    std::reverse(idx_to_remove.begin(), idx_to_remove.end());
    for (auto idx: idx_to_remove)
    {
        /* cout << "removing " << tmp_ind.program.at(idx)->name */ 
        /*     << " at " << idx << "\n"; */
        tmp_ind.program.erase(tmp_ind.program.begin()+idx);
    }
    int end_size = tmp_ind.size();
    logger.log("pattern pruning reduced best model size by " 
            + to_string(starting_size - end_size)
            + " nodes\n=========\n", 2);
    if (tmp_ind.size() < ind.size())
    {
        ind = tmp_ind;
        logger.log("new model:" + this->get_ind_eqn(false, ind),2);
    }

    ///////////////////
    // prune dimensions
    ///////////////////
    /* set_verbosity(3); */
    int iterations = ind.get_dim();
    logger.log("\n=========\ndoing correlation deletion mutations...",2);
    starting_size = ind.size();
    VectorXf original_yhat;
    if (params.classification && params.n_classes==2)
         original_yhat = ind.predict_proba(*d.o).row(0); 
    else
         original_yhat = ind.yhat; 

    for (int i = 0; i < iterations; ++i)
    {
        Individual tmp_ind = ind;
        bool perfect_correlation = variator.correlation_delete_mutate(
                tmp_ind, ind.Phi, params, *d.o);

        if (ind.size() == tmp_ind.size())
        {
            continue;
        }

        tmp_ind.fit(*d.o, params);

        VectorXf new_yhat;
        if (params.classification && params.n_classes==2)
             new_yhat = tmp_ind.predict_proba(*d.o).row(0); 
        else
             new_yhat = tmp_ind.yhat; 


        if (((original_yhat - new_yhat).norm()/original_yhat.norm() 
                <= this->simplify ) 
                or perfect_correlation)
        {
            logger.log("\ndelete dimension mutation success: went from "
                + to_string(ind.size()) + " to " 
                + to_string(tmp_ind.size()) + " nodes. Output changed by " 
                 + to_string(100*(original_yhat
                        -new_yhat).norm()/(original_yhat.norm()))
                 + " %", 2); 
            if (perfect_correlation)
                logger.log("perfect correlation",2);
            ind = tmp_ind;
        }
        else
        {
            logger.log("\ndelete dimension mutation failure. Output changed by " 
                 + to_string(100*(original_yhat
                        -new_yhat).norm()/(original_yhat.norm()))
                 + " %", 2);
            // if this mutation fails, it will continue to fail since it 
            // is deterministic. so, break in this case.
            break;
        }

    }
    end_size = ind.size();
    logger.log("correlation pruning reduced best model size by " 
            + to_string(starting_size - end_size)
            + " nodes\n=========\n", 2);
    if (end_size < starting_size)
        logger.log("new model:" + this->get_ind_eqn(false, ind),2);

    /////////////////
    // prune subtrees
    /////////////////
    iterations = 1000;
    logger.log("\n=========\ndoing subtree deletion mutations...", 2);
    starting_size = ind.size();
    for (int i = 0; i < iterations; ++i)
    {
        Individual tmp_ind = ind;
        this->variator.delete_mutate(tmp_ind, params);
        if (ind.size() == tmp_ind.size())
            continue;

        tmp_ind.fit(*d.o, params);

        VectorXf new_yhat;
        if (params.classification && params.n_classes==2)
             new_yhat = tmp_ind.predict_proba(*d.o).row(0); 
        else
             new_yhat = tmp_ind.yhat; 

        if ((original_yhat - new_yhat).norm()/original_yhat.norm() 
                <= this->simplify )
        {
            logger.log("\ndelete mutation success: went from "
                + to_string(ind.size()) + " to " 
                + to_string(tmp_ind.size()) + " nodes. Output changed by " 
                 + to_string(100*(original_yhat
                        -new_yhat).norm()/(original_yhat.norm()))
                 + " %", 2); 
            ind = tmp_ind;
        }
        else
        {
            logger.log("\ndelete mutation failure. Output changed by " 
                 + to_string(100*(original_yhat
                        -new_yhat).norm()/(original_yhat.norm()))
                 + " %", 2);
            // if this mutation fails, it will continue to fail since it 
            // is deterministic. so, break in this case.
            break;
        }

    }
    end_size = ind.size();
    logger.log("subtree deletion reduced best model size by " 
            + to_string( starting_size - end_size )
            + " nodes", 2);
    VectorXf new_yhat;
    if (params.classification && params.n_classes==2)
         new_yhat = ind.predict_proba(*d.o).row(0); 
    else
         new_yhat = ind.yhat; 
    VectorXf difference = new_yhat - original_yhat;
    /* cout << "final % difference: " << difference.norm()/original_yhat.norm() */ 
    /*     << endl; */
}

vector<float> Feat::univariate_initial_model(DataRef &d, int n_feats) 
{
    /*!
     * If there are more data variables than the max feature size can allow, we 
     * can't initialize a model in the population without some sort of feature 
     * selection. To select features we do the following:
     * 1) fit univariate models to all features in X and store the coefficients
     * 2) fit univariate models to all features in median(Z) and store the 
     * coefficients
     * 3) set terminal weights according to the univariate scores
     * 4) construct a program of dimensionality n_feats using 
     * the largest magnitude coefficients
     */
    vector<float> univariate_weights(d.t->X.rows() + d.t->Z.size(),0.0);
    int N = d.t->X.cols();

    MatrixXf predictor(1,N);    
    string ml_type = this->params.classification? 
        "LR" : "LinearRidgeRegression";
    
    ML ml = ML(ml_type,params.normalize,params.classification,params.n_classes);

    bool pass = true;

    logger.log("univariate_initial_model",2);
    logger.log("N: " + to_string(N),2); 
    logger.log("n_feats: " + to_string(n_feats),2);

    for (unsigned i =0; i<d.t->X.rows(); ++i)
    {
        predictor.row(0) = d.t->X.row(i);
        /* float b =  (covariance(predictor,d.t->y) / */ 
        /*             variance(predictor)); */
        pass = true;
        shared_ptr<CLabels> yhat = ml.fit(predictor, d.t->y, this->params, 
                pass);
        if (pass)
            univariate_weights.at(i) = ml.get_weights().at(0);
        else
            univariate_weights.at(i) = 0;
    }
    int j = d.t->X.rows();
    for (const auto& val: d.t->Z)
    {
        for (int k = 0; k<N; ++k)
            predictor(k) = median(val.second.second.at(k));

        /* float b =  (covariance(predictor,d.t->y) / */ 
        /*             variance(predictor)); */
        /* univariate_weights.at(j) = fabs(b); */

        pass = true;
        shared_ptr<CLabels> yhat = ml.fit(predictor, d.t->y, this->params, 
                pass);
        if (pass)
            univariate_weights.at(j) = ml.get_weights().at(0);
        else
            univariate_weights.at(j) = 0;

        ++j;
    }
    return univariate_weights;

}
void Feat::initial_model(DataRef &d)
{
    /*!
     * fits an ML model to the raw data as a starting point.
     */
     
    best_ind = Individual();
    best_ind.set_id(0);
    int j; 
    int n_x = d.t->X.rows();
    int n_z = d.t->Z.size();
    int n_feats = std::min(params.max_dim, unsigned(n_x+ n_z));
    /* int n_long_feats = std::min(params.max_dim - n_feats, */ 
    /*         unsigned(d.t->Z.size())); */
    bool univariate_initialization = false;

    if (n_feats < (n_x + n_z))
    {
        // if the data has more features than params.max_dim, fit a univariate 
        // linear model to each feature in order to set initial weights
        univariate_initialization = true;
        vector<float> univariate_weights = univariate_initial_model(d, 
                n_feats);
        
        vector<size_t> feature_order = argsort(univariate_weights, false);
        feature_order.erase(feature_order.begin()+n_feats,
                            feature_order.end());

        for (const auto& f : feature_order)
        {
            if (f < n_x)
                best_ind.program.push_back(params.terminals.at(f)->clone());
            else
            {
                best_ind.program.push_back(params.terminals.at(f)->clone());
                best_ind.program.push_back(
                        std::unique_ptr<Node>(new NodeMedian()));
            }

        }
        params.set_term_weights(univariate_weights);
    }
    else
    {
        for (unsigned i =0; i<n_x; ++i)
        {
            best_ind.program.push_back(params.terminals.at(i)->clone());
        }
        // if there is longitudinal data, initialize the model with median 
        // values applied to those variables.
        for (unsigned i =0; i<n_z; ++i)
        {
            best_ind.program.push_back(params.terminals.at(n_x + i)->clone());
            best_ind.program.push_back(
                    std::unique_ptr<Node>(new NodeMedian()));
        }
    }
    // fit model

    shared_ptr<CLabels> yhat;

    
    if (univariate_initialization)
    { 
        yhat = best_ind.fit(*d.t,params);
    }
    else 
    {
        // tune default ML parameters 
        if (params.tune_initial)
            yhat = best_ind.fit_tune(*d.t, params, true);
        else
            yhat = best_ind.fit(*d.t, params);
        // set terminal weights based on model
        vector<float> w = best_ind.ml->get_weights();

        params.set_term_weights(w);
    }

    this->min_loss = evaluator.S.score(d.t->y, yhat, params.class_weights);
    
    if (params.split < 1.0)
    {
        shared_ptr<CLabels> yhat_v = best_ind.predict(*d.v);
        this->min_loss_v = evaluator.S.score(d.v->y, yhat_v, 
                                             params.class_weights); 
    }
    else
        this->min_loss_v = min_loss;
    
    best_ind.fitness = min_loss;
    
    this->best_complexity = best_ind.get_complexity();

    logger.log("initial model: " + this->get_eqn(), 2);
    logger.log("initial training score: " +std::to_string(min_loss),2);
    logger.log("initial validation score: " +std::to_string(this->min_loss_v),2);
}

MatrixXf Feat::transform(MatrixXf& X)
{
    LongData Z;
    return transform(X,Z);
}
MatrixXf Feat::transform(MatrixXf& X, LongData& Z)
{
    return transform(X,Z,nullptr);
}
MatrixXf Feat::transform(MatrixXf& X,
                         LongData Z,
                         Individual *ind)
{
    /*!
     * Transforms input data according to ind or best ind, if ind is undefined.
     */
    
    if (params.normalize)
        N.normalize(X);       
    
    VectorXf y = VectorXf();
    
    Data d(X, y, Z, get_classification());
    
    if (ind == 0)        // if ind is empty, predict with best_ind
    {
        if (best_ind.program.size()==0)
            THROW_RUNTIME_ERROR("You need to train a model using fit() "
                    "before making predictions.");
        
        return best_ind.out(d, true).transpose();
    }

    return ind->out(d, true).transpose();
}

VectorXf Feat::predict(MatrixXf& X)
{
    auto Z = LongData();
    return predict(X,Z);
}

VectorXf Feat::predict(MatrixXf& X,
                       LongData& Z)
{        
    /* MatrixXf Phi = transform(X, Z); */
    if (params.normalize)
        N.normalize(X);       
    VectorXf dummy;
    Data d_tmp(X, dummy, Z);
    return best_ind.predict_vector(d_tmp);
}

VectorXf Feat::predict_archive(int id, MatrixXf& X)
{
    LongData Z; 
    return predict_archive(id, X, Z);
}

VectorXf Feat::predict_archive(int id, MatrixXf& X, LongData& Z)
{
    /* cout << "Feat::predict_archive\n"; */
    /* return predictions; */
    /* cout << "Normalize" << endl; */
    if (params.normalize)
        N.normalize(X);       
    /* cout << "params.n_classes:" << params.n_classes << endl; */
    /* cout << "X.cols(): " << X.cols() << endl; */
    VectorXf predictions(X.cols());
    VectorXf empty_y;
    /* cout << "tmp_data\n"; */ 
    Data tmp_data(X,empty_y,Z);

    /* cout << "individual prediction id " << id << "\n"; */
    if (id == best_ind.id)
    {
        return best_ind.predict_vector(tmp_data);
    }
    for (int i = 0; i < this->archive.individuals.size(); ++i)
    {
        Individual& ind = this->archive.individuals.at(i);

        if (id == ind.id)
            return ind.predict_vector(tmp_data);

    }
    for (int i = 0; i < this->pop.individuals.size(); ++i)
    {
        Individual& ind = this->pop.individuals.at(i);

        if (id == ind.id)
            return ind.predict_vector(tmp_data);

    }

    THROW_INVALID_ARGUMENT("Could not find id = "
            + to_string(id) + "in archive or population.");
    return VectorXf();
}

ArrayXXf Feat::predict_proba_archive(int id, MatrixXf& X)
{
    LongData Z;
    return predict_proba_archive(id, X, Z);
}
ArrayXXf Feat::predict_proba_archive(int id, MatrixXf& X, LongData& Z)
{
    if (params.normalize)
        N.normalize(X);       
    ArrayXXf predictions(X.cols(),params.n_classes);
    VectorXf empty_y;
    Data tmp_data(X,empty_y,Z);

    for (int i = 0; i < this->archive.individuals.size(); ++i)
    {
        Individual& ind = this->archive.individuals.at(i);

        if (id == ind.id)
            return ind.predict_proba(tmp_data);

    }

    THROW_INVALID_ARGUMENT("Could not find id = "
            + to_string(id) + "in archive.");
    return ArrayXXf();
    
}
shared_ptr<CLabels> Feat::predict_labels(MatrixXf& X, LongData Z)
{        
    /* MatrixXf Phi = transform(X, Z); */
    if (params.normalize)
        N.normalize(X);       
    VectorXf empty_y;
    Data tmp_data(X,empty_y,Z);

    return best_ind.predict(tmp_data);        
}

ArrayXXf Feat::predict_proba(MatrixXf& X, LongData& Z)
{
    if (params.normalize)
        N.normalize(X);       
    VectorXf dummy;
    Data d_tmp(X, dummy, Z);
    return best_ind.predict_proba(d_tmp);
}

ArrayXXf Feat::predict_proba(MatrixXf& X)
{
    LongData Z;
    return predict_proba(X,Z);
}


bool Feat::update_best(const DataRef& d, bool val)
{
    float bs;
    bs = this->min_loss_v; 
    float f; 
    vector<Individual>& pop_ref = (use_arch ? 
                               archive.individuals : this->pop.individuals); 

    bool updated = false; 

    for (const auto& ind: pop_ref)
    {
        if (!val_from_arch || ind.rank == 1)
        {
            f = ind.fitness_v;

            if (f < bs 
                || (f == bs && ind.get_complexity() < this->best_complexity)
                )
            {
                bs = f;
                this->best_ind = ind; // should this be ind.clone(best_ind); ?
                /* ind.clone(best_ind); */
                this->best_complexity = ind.get_complexity();
                updated = true;
                logger.log("better model found!", 2);
            }
        }
    }
    logger.log("current best model: " + this->get_eqn(), 2);
    this->min_loss_v = bs; 

    return updated;
}

float Feat::score(MatrixXf& X, const VectorXf& y, LongData Z)
{
    shared_ptr<CLabels> labels = predict_labels(X, Z);
    VectorXf loss; 
    return evaluator.S.score(y,labels,loss,params.class_weights);
}

void Feat::calculate_stats(const DataRef& d)
{

    VectorXf losses(this->pop.size());
    int i=0;
    for (const auto& p: this->pop.individuals)
    {
        losses(i) = p.fitness;
        ++i;
    }
    // min loss
    float min_loss = losses.minCoeff();  

    // median loss
    float med_loss = median(losses.array());  
    
    // median program size
    ArrayXf Sizes(this->pop.size());
    
    i = 0;
    
    for (const auto& p : this->pop.individuals)
    { 
        Sizes(i) = p.size(); 
        ++i;
    }
    unsigned med_size = median(Sizes);                        
    
    // complexity
    ArrayXf Complexities(this->pop.size()); 
    i = 0; 
    for (auto& p : this->pop.individuals)
    {
        // Calculate to assure it gets reported in stats (even if's not used as an obj)
        Complexities(i) = p.get_complexity(); 
        ++i;
    }

    // number of parameters
    ArrayXf Nparams(this->pop.size()); 
    i = 0; 
    for (auto& p : this->pop.individuals)
    { 
        Nparams(i) = p.get_n_params(); 
        ++i;
    }

    // dimensions
    ArrayXf Dims(this->pop.size()); 
    i = 0; 
    for (auto& p : this->pop.individuals)
    { 
        Dims(i) = p.get_dim(); 
        ++i;
    }
    
    /* unsigned med_size = median(Sizes); */ 
    unsigned med_complexity = median(Complexities);            
    unsigned med_num_params = median(Nparams);                
    unsigned med_dim = median(Dims);                          
    
    // calculate the median valiation loss 
    ArrayXf val_fitnesses(this->pop.individuals.size());
    for (unsigned i = 0; i < this->pop.individuals.size(); ++i)
        val_fitnesses(i) = this->pop.individuals.at(i).fitness_v;
    float med_loss_v = median(val_fitnesses); 
        /* fitnesses.push_back(pop.individuals.at(i).fitness); */
    /* int idx = argmiddle(fitnesses); */

    /* if (params.split < 1.0) */
    /* { */
        /* Individual& med_ind = pop.individuals.at(idx); */
        /* VectorXf tmp; */
        /* shared_ptr<CLabels> yhat_v = med_ind.predict(*d.v, params); */
        /* this->med_loss_v = p_eval->S.score(d.v->y, yhat_v, tmp, */ 
        /*         params.class_weights); */ 
    /* } */
    
    /* ///////////////////////////////////////////// */
   
    // update stats
    stats.update(params.current_gen,
                 timer.Elapsed().count(),
                 min_loss,
                 this->min_loss_v,
                 med_loss,
                 med_loss_v,
                 med_size,
                 med_complexity,
                 med_num_params,
                 med_dim);
}

void Feat::print_stats(std::ofstream& log, float fraction)
{
    unsigned num_models = std::min(50,this->pop.size());
    //float med_loss = median(F.colwise().mean().array());  // median loss
    // collect program sizes
    ArrayXf Sizes(this->pop.size()); 
    unsigned i = 0;               
    for (const auto& p : this->pop.individuals)
    { 
        Sizes(i) = p.size(); ++i;
    }
    unsigned max_size = Sizes.maxCoeff();
    // progress bar
    string bar, space = "";                                 
    for (unsigned int i = 0; i<50; ++i)
    {
        if (i <= 50*fraction) bar += "/";
        else space += " ";
    }
    std::cout.precision(5);
    std::cout << std::scientific;
    
    if(params.max_time == -1)
        std::cout << "Generation " << params.current_gen+1 << "/" 
            << params.gens << " [" + bar + space + "]\n";
    else
        std::cout << std::fixed << "Time elapsed "<< timer 
            << "/" << params.max_time 
            << " seconds (Generation "<< params.current_gen+1 
            << ") [" + bar + space + "]\n";
        
    std::cout << std::fixed << "Train Loss (Med): " 
              << stats.min_loss.back() << " (" 
              << stats.med_loss.back() << ")\n"
              << "Val Loss (Med): " 
              << stats.min_loss_v.back() << " (" << stats.med_loss_v.back() << ")\n"
              << "Median Size (Max): " 
              << stats.med_size.back() << " (" << max_size << ")\n"
              << "Time (s): "   << timer << "\n";
    std::cout << "Representation Pareto Front--------------------------------------\n";
    std::cout << "Rank\t"; //Complexity\tLoss\tRepresentation\n";
    /* for (const auto& o : params.objectives) */
    /*     std::cout << o << "\t"; */
    cout << "fitness\tfitness_v\tcomplexity\t";
    cout << "Representation\n";

    std::cout << std::scientific;
    // printing max 40 individuals from the pareto front
    unsigned n = 1;
    if (use_arch)
    {
        num_models = std::min(40, int(archive.individuals.size()));

        for (unsigned i = 0; i < num_models; ++i)
        {
            std::string lim_model;

            std::string model = this->get_ind_eqn(false, archive.individuals[i]);
            /* std::string model = archive.individuals[i].get_eqn(); */
            for (unsigned j = 0; j< std::min(model.size(),size_t(60)); ++j)
            {
                lim_model.push_back(model.at(j));
            }
            if (lim_model.size()==60) 
                lim_model += "...";
            
            std::cout <<  archive.individuals[i].rank          << "\t" 
            /* for (const auto& o : archive.individuals[i].obj) */
            /*     std::cout << o << "\t"; */
                  <<  archive.individuals[i].fitness       << "\t" 
                  <<  archive.individuals[i].fitness_v       << "\t" 
                  <<  archive.individuals[i].get_complexity()  << "\t" ;
            cout <<  lim_model << "\n";  
        }
    }
    else
    {
        vector<size_t> f = this->pop.sorted_front(n);
        vector<size_t> fnew(2,0);
        while (f.size() < num_models && fnew.size()>1)
        {
            fnew = this->pop.sorted_front(++n);                
            f.insert(f.end(),fnew.begin(),fnew.end());
        }
        
        for (unsigned j = 0; j < std::min(num_models,unsigned(f.size())); ++j)
        {     
            std::string lim_model;
            std::string model = this->get_ind_eqn(false,pop.individuals[f[j]]);
            /* std::string model = this->pop.individuals[f[j]].get_eqn(); */
            for (unsigned j = 0; j< std::min(model.size(),size_t(60)); ++j)
                lim_model.push_back(model.at(j));
            if (lim_model.size()==60) 
                lim_model += "...";
            std::cout << pop.individuals[f[j]].rank              << "\t" 
                      << pop.individuals[f[j]].fitness              << "\t" 
                      << pop.individuals[f[j]].fitness_v              << "\t" 
                      << pop.individuals[f[j]].get_complexity()              << "\t" ;
            cout << "\t" << lim_model << "\n";  
        }
    }
   
    std::cout <<"\n\n";
}

void Feat::log_stats(std::ofstream& log)
{
    // print stats in tabular format
    string sep = ",";
    if (params.current_gen == 0) // print header
    {
        log << "generation"     << sep
            << "time"           << sep
            << "min_loss"       << sep 
            << "min_loss_val"   << sep 
            << "med_loss"       << sep 
            << "med_loss_val"   << sep 
            << "med_size"       << sep 
            << "med_complexity" << sep 
            << "med_num_params" << sep
            << "med_dim"        << "\n";
    }
    log << params.current_gen          << sep
        << timer.Elapsed().count()     << sep
        << stats.min_loss.back()       << sep
        << stats.min_loss_v.back()     << sep
        << stats.med_loss.back()       << sep
        << stats.med_loss_v.back()     << sep
        << stats.med_size.back()       << sep
        << stats.med_complexity.back() << sep
        << stats.med_num_params.back() << sep
        << stats.med_dim.back()        << "\n"; 
}

//TODO: replace these with json
json Feat::get_stats()
{ 
    json j; 
    to_json(j, this->stats); 
    return j;
}

void Feat::load_best_ind(string filename)
{
    //TODO: need to load/save normalizer
    this->best_ind.load(filename);
}

void Feat::load_population(string filename, bool justfront)
{
    this->pop.load(filename);
}

void Feat::load(const json& j)
{
    // json j = json::parse(feat_state);
    from_json(j, *this);
}

json Feat::save() const
{
    json j;
    to_json(j, *this);
    return j;
}

void Feat::load_from_file(string filename)
{
    std::ifstream indata;
    indata.open(filename);
    if (!indata.good())
        THROW_INVALID_ARGUMENT("Invalid input file " + filename + "\n"); 

    std::string line;
    indata >> line; 

    this->load(line);

    logger.log("Loaded Feat state from " + filename,1);

    indata.close();
}

void Feat::save_to_file(string filename)
{
    std::ofstream out;                      
    if (!filename.empty())
        out.open(filename);
    else
        out.open("Feat.json");

    out << this->save();
    out.close();
    logger.log("Saved Feat to file " + filename, 1);
}
