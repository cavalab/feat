/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef FEWTWO_H
#define FEWTWO_H

//external includes
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <shogun/base/init.h>
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif
// stuff being used
using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::cout; 
 
// internal includes
#include "rnd.h"
#include "utils.h"
#include "params.h"
#include "population.h"
#include "selection.h"
#include "evaluation.h"
#include "variation.h"
#include "ml.h"
#include "node/node.h"
 

//shogun initialization
void __attribute__ ((constructor)) ctor()
{
    //cout<< "INITIALIZING SHOGUN\n";
    init_shogun_with_defaults();
}

void __attribute__ ((destructor))  dtor()
{
    //cout<< "EXITING SHOGUN\n";
    exit_shogun();
}

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Feat
     * @brief main class for the Feat learner.
     *   
     * @details Feat optimizes feature represenations for a given machine learning algorithm. It 
     *			does so by using evolutionary computation to optimize a population of programs. 
     *			Each program represents a set of feature transformations. 
     */
    class Feat 
    {
        public : 
                        
            // Methods 
            
            /// member initializer list constructor
              
            Feat(int pop_size=100, int gens = 100, string ml = "LinearRidgeRegression", 
                   bool classification = false, int verbosity = 1, int max_stall = 0,
                   string sel ="lexicase", string surv="pareto", float cross_rate = 0.5,
                   char otype='a', string functions = "+,-,*,/,^2,^3,exp,log,and,or,not,=,<,>,ite", 
                   unsigned int max_depth = 3, unsigned int max_dim = 10, int random_state=0, 
                   bool erc = false, string obj="fitness,complexity",bool shuffle=false, 
                   double split=0.75, double fb=0.5):
                      // construct subclasses
                      params(pop_size, gens, ml, classification, max_stall, otype, verbosity, 
                             functions, cross_rate, max_depth, max_dim, erc, obj, shuffle, split, fb), 
                      p_sel( make_shared<Selection>(sel) ),
                      p_surv( make_shared<Selection>(surv, true) ),
                      p_eval( make_shared<Evaluation>() ),
                      p_variation( make_shared<Variation>(cross_rate) )                      
            {
                r.set_seed(random_state);
                str_dim = "";
                name="";
            }
            
            /// set size of population 
            void set_pop_size(int pop_size){ params.pop_size = pop_size; }            
            
            /// set size of max generations              
            void set_generations(int gens){ params.gens = gens; }         
                        
            /// set ML algorithm to use              
            void set_ml(string ml){ params.ml = ml; }            
            
            /// set EProblemType for shogun              
            void set_classification(bool classification){ params.classification = classification;}
                 
            /// set level of debug info              
            void set_verbosity(int verbosity){ params.set_verbosity(verbosity); }
                        
            /// set maximum stall in learning, in generations
            void set_max_stall(int max_stall){	params.max_stall = max_stall; }
                        
            /// set selection method              
            void set_selection(string sel){ p_sel = make_shared<Selection>(sel); }
                        
            /// set survivability              
            void set_survival(string surv){ p_surv = make_shared<Selection>(surv, true); }
                        
            /// set cross rate in variation              
            void set_cross_rate(float cross_rate){ params.cross_rate = cross_rate; p_variation->set_cross_rate(cross_rate); }
                        
            /// set program output type ('f', 'b')              
            void set_otype(char ot){ params.set_otype(ot); }
                        
            /// sets available functions based on comma-separated list.
            void set_functions(string functions){ params.set_functions(functions); }
                        
            /// set max depth of programs              
            void set_max_depth(unsigned int max_depth){ params.set_max_depth(max_depth); }
             
            /// set maximum dimensionality of programs              
            void set_max_dim(unsigned int max_dim){	params.set_max_dim(max_dim); }
            
            ///set dimensionality as multiple of the number of columns
            void set_max_dim(string str) { str_dim = str; }            
            
            /// set seeds for each core's random number generator              
            void set_random_state(int random_state){ r.set_seed(random_state); }
                        
            /// flag to set whether to use variable or constants for terminals              
            void set_erc(bool erc){ params.erc = erc; }
            
            /// flag to shuffle the input samples for train/test splits
            void set_shuffle(bool sh){params.shuffle = sh;}

            /// set objectives in feat
            void set_objectives(string obj){ params.set_objectives(obj); }
            /// set train fraction of dataset
            void set_split(double sp){params.split = sp;}
            
            ///set data types for input parameters
            void set_dtypes(vector<char> dtypes){params.dtypes = dtypes;}

            ///set feedback
            void set_feedback(double fb){ params.feedback = fb;}

            ///set name for files
            void set_name(string s){name = s;}

            /*                                                      
             * getting functions
             */

            ///return population size
            int get_pop_size(){ return params.pop_size; }
            
            ///return size of max generations
            int get_generations(){ return params.gens; }
            
            ///return ML algorithm string
            string get_ml(){ return params.ml; }
            
            ///return type of classification flag set
            bool get_classification(){ return params.classification; }
            
            ///return maximum stall in learning, in generations
            int get_max_stall() { return params.max_stall; }
            
            ///return program output type ('f', 'b')             
            vector<char> get_otypes(){ return params.otypes; }
            
            ///return current verbosity level set
            int get_verbosity(){ return params.verbosity; }
            
            ///return max_depth of programs
            int get_max_depth(){ return params.max_depth; }
            
            ///return cross rate for variation
            float get_cross_rate(){ return params.cross_rate; }
            
            ///return max size of programs
            int get_max_size(){ return params.max_size; }
            
            ///return max dimensionality of programs
            int get_max_dim(){ return params.max_dim; }
            
            ///return boolean value of erc flag
            bool get_erc(){ return params.erc; }
           
            /// get name
            string get_name(){ return name; }

            ///return number of features
            int get_num_features(){ return params.num_features; }
            
            ///return whether option to shuffle the data is set or not
            bool get_shuffle(){ return params.shuffle; }
            
            ///return fraction of data to use for training
            double get_split(){ return params.split; }
            
            ///add custom node into feat
            void add_function(shared_ptr<Node> N){ params.functions.push_back(N); }
            
            ///return data types for input parameters
            vector<char> get_dtypes(){ return params.dtypes; }

            ///return feedback setting
            double get_feedback(){ return params.feedback; }
            
            ///return population as string
            string get_eqns(bool front=true)
            {
                string r="complexity,fitness,eqn\n";
                if (front)  // only return individuals on the Pareto front
                {
                    // printing individuals from the pareto front
                    unsigned n = 1;
                    vector<size_t> f = p_pop->sorted_front(n);
                   
                    
                    for (unsigned j = 0; j < f.size(); ++j)
                    {          
                        r += std::to_string(p_pop->individuals[f[j]].complexity()) + "," 
                            + std::to_string((*p_pop)[f[j]].fitness) + "," 
                            + p_pop->individuals[f[j]].get_eqn() + "\n";  
                    }
                }
                else
                {
                    for (unsigned j = 0; j < params.pop_size; ++j)
                    {          
                        r += std::to_string(p_pop->individuals[j].complexity()) + "," 
                            + std::to_string((*p_pop)[j].fitness) + "," 
                            + p_pop->individuals[j].get_eqn() + "\n";  
                    }
                }
                return r;
            }
            /// destructor             
            ~Feat(){} 
                        
            /// train a model.             
            void fit(MatrixXd& X, VectorXd& y);

            /// train a model.             
            void fit(double * X,int rowsX,int colsX, double * Y,int lenY);
            
            /// predict on unseen data.             
            VectorXd predict(MatrixXd& X);    

            /// predict on unseen data.             
            VectorXd predict(double * X, int rowsX, int colsX);     
            
            /// transform an input matrix using a program.                          
            MatrixXd transform(MatrixXd& X,  Individual *ind = 0);
            
	    MatrixXd transform(double * X,  int rows_x, int cols_x);
            
            /// convenience function calls fit then predict.            
            VectorXd fit_predict(MatrixXd& X, VectorXd& y){ fit(X,y); return predict(X); } 
           
            VectorXd fit_predict(double * X, int rows_x, int cols_x, double * Y, int len_y)
		{
			MatrixXd matX = Map<MatrixXd>(X,rows_x,cols_x);
			VectorXd vectY = Map<VectorXd>(Y,len_y);
			fit(matX,vectY); 
			return predict(matX); 
		} 
            
            /// convenience function calls fit then transform. 
            MatrixXd fit_transform(MatrixXd& X, VectorXd& y){ fit(X,y); return transform(X); }

            MatrixXd fit_transform(double * X, int rows_x, int cols_x, double * Y, int len_y)
		{
			MatrixXd matX = Map<MatrixXd>(X,rows_x,cols_x);
			VectorXd vectY = Map<VectorXd>(Y,len_y);
			fit(matX,vectY); 
			return transform(matX);
		}
                  
            /// scoring function 
            double score(MatrixXd& X, VectorXd& y);
            
        private:
            // Parameters
            Parameters params;    					///< hyperparameters of Feat 
            MatrixXd F;                 			///< matrix of fitness values for population
            MatrixXd F_v;                           ///< matrix of validation scores
            Timer timer;                            ///< start time of training
            string name;                            ///< name to append to files
            // subclasses for main steps of the evolutionary computation routine
            shared_ptr<Population> p_pop;       	///< population of programs
            shared_ptr<Selection> p_sel;        	///< selection algorithm
            shared_ptr<Evaluation> p_eval;      	///< evaluation code
            shared_ptr<Variation> p_variation;  	///< variation operators
            shared_ptr<Selection> p_surv;       	///< survival algorithm
            shared_ptr<ML> p_ml;                	///< pointer to machine learning class
            // performance tracking
            double best_score;                      ///< current best score
            string str_dim;                         ///< dimensionality as multiple of number of columns 
            void update_best();                     ///< updates best score   
            void print_stats(unsigned int);         ///< prints stats
            Individual best_ind;                    ///< best individual
            /// method to fit inital ml model            
            void initial_model(MatrixXd& X, VectorXd& y);
            /// fits final model to best transformation
            void final_model(MatrixXd& X, VectorXd& y);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Feat::fit(MatrixXd& X, VectorXd& y)
    {
        /*!
         *  Input:
         
         *       X: n_features x n_samples MatrixXd of features
         *       y: VectorXd of labels 
         
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
        // start the clock
        timer.Reset();
        
        if(str_dim.compare("") != 0)
        {
            string dimension;
            dimension = str_dim.substr(0, str_dim.length() - 1);
            params.msg("STR DIM IS "+ dimension, 1);
            params.msg("Cols are " + std::to_string(X.rows()), 1);
            params.msg("Setting dimensionality as " + 
                       std::to_string((int)(ceil(stod(dimension)*X.rows()))), 1);
            set_max_dim(ceil(stod(dimension)*X.rows()));
        }
        
        params.check_ml();       

        if (params.classification)  // setup classification endpoint
            params.set_classes(y);
         
        set_dtypes(find_dtypes(X));

        p_ml = make_shared<ML>(params); // intialize ML
        p_pop = make_shared<Population>(params.pop_size);

        // split data into training and test sets
        MatrixXd X_t(X.rows(),int(X.cols()*params.split));
        MatrixXd X_v(X.rows(),int(X.cols()*(1-params.split)));
        VectorXd y_t(int(y.size()*params.split)), y_v(int(y.size()*(1-params.split)));
        train_test_split(X,y,X_t,X_v,y_t,y_v,params.shuffle);
        
        // define terminals based on size of X
        params.set_terminals(X.rows());        

        // initial model on raw input
        params.msg("Fitting initial model", 1);
        initial_model(X_t,y_t);  
        params.msg("Initial score: " + std::to_string(best_score), 1);
        
        // initialize population 
        params.msg("Initializing population", 1);
        p_pop->init(best_ind,params);
        params.msg("Initial population:\n"+p_pop->print_eqns(),2);

        // resize F to be twice the pop-size x number of samples
        F.resize(X_t.cols(),int(2*params.pop_size));
       
        // evaluate initial population
        params.msg("Evaluating initial population",1);
        p_eval->fitness(*p_pop,X_t,y_t,F,params);

        vector<size_t> survivors;

        // main generational loop
        for (unsigned int g = 0; g<params.gens; ++g)
        {        
            // select parents
            params.msg("selection..", 2);
            vector<size_t> parents = p_sel->select(*p_pop, F, params);
            params.msg("parents:\n"+p_pop->print_eqns(), 2);          
            
            // variation to produce offspring
            params.msg("variation...", 2);
            p_variation->vary(*p_pop, parents, params);
            params.msg("offspring:\n" + p_pop->print_eqns(true), 2);

            // evaluate offspring
            params.msg("evaluating offspring...", 2);
            p_eval->fitness(*p_pop, X_t, y_t, F, params, true);

            // select survivors from combined pool of parents and offspring
            params.msg("survival...", 2);
            survivors = p_surv->survive(*p_pop, F, params);
           
            // reduce population to survivors
            params.msg("shrinking pop to survivors...",2);
            p_pop->update(survivors);
            params.msg("survivors:\n" + p_pop->print_eqns(), 2);

            update_best();
            if (params.verbosity>0) print_stats(g+1);           
        }
        params.msg("finished",1);
        params.msg("best training representation: " + best_ind.get_eqn(),1);
        params.msg("train score: " + std::to_string(best_score), 1);
        // evaluate population on validation set
        if (params.split < 1.0)
        {
            F_v.resize(X_v.cols(),int(2*params.pop_size)); 
            p_eval->fitness(*p_pop, X_v, y_v, F_v, params);
            initial_model(X_v, y_v);        // calculate baseline model validation score
            update_best();                  // get the best validation model
        }
        
        final_model(X,y);   // fit final model to best model
        params.msg("best validation representation: " + best_ind.get_eqn(),1);
        params.msg("validation score: " + std::to_string(best_score), 1);

        
        // write model to file
        std::ofstream out_model; 
        out_model.open("model_" + name + ".txt");
        out_model << best_ind.get_eqn() ; 
        out_model.close();
    }

    void Feat::fit(double * X, int rowsX, int colsX, double * Y, int lenY)
    {
        MatrixXd matX = Map<MatrixXd>(X,rowsX,colsX);
        VectorXd vectY = Map<VectorXd>(Y,lenY);
	
	Feat::fit(matX,vectY);
    }

    void Feat::final_model(MatrixXd& X, VectorXd& y)
    {
        // fits final model to best tranformation found.
        bool pass = true;
        MatrixXd Phi = transform(X);
        VectorXd yhat = p_ml->out(Phi,y,params,pass,best_ind.dtypes);
    }
    void Feat::initial_model(MatrixXd& X, VectorXd& y)
    {
        /*!
         * fits an ML model to the raw data as a starting point.
         */
        bool pass = true;
        VectorXd yhat = p_ml->out(X,y,params,pass);
 
        // set terminal weights based on model
        params.set_term_weights(p_ml->get_weights());

        if (params.classification)  // assign best score as balanced accuracy
            best_score = p_eval->bal_accuracy(y, yhat, params.classes);
        else                        // assign best score as MSE
            best_score = p_eval->se(y,yhat).mean();
        
        // initialize best_ind to be all the features
        best_ind = Individual();
        for (unsigned i =0; i<X.rows(); ++i)
            best_ind.program.push_back(params.terminals[i]);
        best_ind.fitness = best_score;
    }

    MatrixXd Feat::transform(MatrixXd& X, Individual *ind)
    {
        /*!
         * Transforms input data according to ind or best ind, if ind is undefined.
         */
        if (ind == 0)        // if ind is empty, predict with best_ind
        {
            if (best_ind.program.size()==0){
                std::cerr << "You need to train a model using fit() before making predictions.\n";
                throw;
            }
            normalize(X,params.dtypes);
            MatrixXd Phi = best_ind.out(X,params);
            normalize(Phi,best_ind.dtypes);
            return Phi;
        }
        normalize(X,params.dtypes);
        MatrixXd Phi = ind->out(X,params);
        normalize(Phi,ind->dtypes);
        return Phi;
    }

    MatrixXd Feat::transform(double * X, int rows_x,int cols_x)
    {
        MatrixXd matX = Map<MatrixXd>(X,rows_x,cols_x);
        return transform(matX);
        
    }
    
    
    VectorXd Feat::predict(MatrixXd& X)
    {        
        MatrixXd Phi = transform(X);
        auto PhiSG = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(Phi));
        SGVector<double> y_pred;
        VectorXd yhat;

        if (params.classification && params.n_classes == 2 && 
                (!params.ml.compare("SVM") || !params.ml.compare("LR")))
        {
            auto tmp = p_ml->p_est->apply_binary(PhiSG);
            y_pred = tmp->get_labels();
            delete tmp;
            
        }
        else if (params.classification)
        {
            auto tmp = p_ml->p_est->apply_multiclass(PhiSG);
            y_pred = tmp->get_labels();
            delete tmp;
        }
        else
        {
            auto tmp = p_ml->p_est->apply_regression(PhiSG);
            y_pred = tmp->get_labels();
            delete tmp;
        }
        
        yhat = Eigen::Map<VectorXd>(y_pred.data(),y_pred.size());
        
        if (params.classification && (!params.ml.compare("LR") || !params.ml.compare("SVM")))
            // convert -1 to 0
            yhat = (yhat.cast<int>().array() == -1).select(0,yhat);
        
        return yhat;        
    }

    VectorXd Feat::predict(double * X, int rowsX,int colsX)
    {
        MatrixXd matX = Map<MatrixXd>(X,rowsX,colsX);
        return Feat::predict(matX);
    }

    void Feat::update_best()
    {
        for (const auto& i: p_pop->individuals)
        {
            if (i.fitness < best_score)
            {
                best_score = i.fitness;
                best_ind = i;
            }
        }
 
    }
    
    double Feat::score(MatrixXd& X, VectorXd& y)
    {
        VectorXd yhat = predict(X);
        
        if (params.classification)
            return p_eval->bal_accuracy(y,yhat,vector<int>(),false);
        else
            return p_eval->se(y,yhat).mean();
    }
    void Feat::print_stats(unsigned int g)
    {
        unsigned num_models = std::min(100,p_pop->size());
        double med_score = median(F.colwise().mean().array());  // median loss
        ArrayXd Sizes(p_pop->size()); unsigned i = 0;           // collect program sizes
        for (const auto& p : p_pop->individuals){ Sizes(i) = p.size(); ++i;}
        unsigned med_size = median(Sizes);                        // median program size
        unsigned max_size = Sizes.maxCoeff();
        string bar, space = "";                                 // progress bar
        for (unsigned int i = 0; i<50; ++i){
            if (i <= 50*g/params.gens) bar += "/";
            else space += " ";
        }
        std::cout.precision(5);
        std::cout << std::scientific;
        std::cout << "Generation " << g << "/" << params.gens << " [" + bar + space + "]\n";
        std::cout << "Min Loss\tMedian Loss\tMedian (Max) Size\tTime (s)\n"
                  <<  best_score << "\t" << med_score << "\t" ;
        std::cout << std::fixed  << med_size << " (" << max_size << ") \t\t" << timer << "\n";
        std::cout << "Representation Pareto Front--------------------------------------\n";
        std::cout << "Rank\tComplexity\tLoss\tRepresentation\n";
        std::cout << std::scientific;
        // printing 10 individuals from the pareto front
        unsigned n = 1;
        vector<size_t> f = p_pop->sorted_front(n);
        vector<size_t> fnew(2,0);
        while (f.size() < num_models && fnew.size()>1)
        {
            fnew = p_pop->sorted_front(++n);                
            f.insert(f.end(),fnew.begin(),fnew.end());
        }
        
        for (unsigned j = 0; j < std::min(num_models,unsigned(f.size())); ++j)
        {          
            std::cout << p_pop->individuals[f[j]].rank << "\t" 
                      <<  p_pop->individuals[f[j]].complexity() << "\t" << (*p_pop)[f[j]].fitness 
                      << "\t" << p_pop->individuals[f[j]].get_eqn() << "\n";  
        }
       
       
        // ref counting
        vector<float> use(params.terminals.size());
        float use_sum=0;
        for (unsigned i = 0; i< params.terminals.size(); ++i)
        {    
            use[i] = float(params.terminals[i].use_count());
            use_sum += use[i];
        }
        vector<size_t> use_idx = argsort(use);
        std::reverse(use_idx.begin(), use_idx.end());

        int nf = std::min(5,int(params.terminals.size()));
        std::cout << "Top " << nf <<" features (\% usage):\n";
        std::cout.precision(1);
        for (unsigned i = 0; i<nf; ++i) 
            std::cout << std::fixed << params.terminals[use_idx[i]]->name  
                      << " (" << use[use_idx[i]]/use_sum*100 << "\%)\t"; 
        
        std::cout <<"\n\n";
    }
    
}
#endif
