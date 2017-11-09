/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef POPULATION_H
#define POPULATION_H

//#include "node.h" // including node.h since definition of node is in the header
#include "individual.h"
using std::vector;
using std::string;
using Eigen::Map;

namespace FT{    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    extern Rnd r;

    /*!
     * @class Population
     */
    struct Population
    {
        vector<Individual> individuals;     ///< individual programs
        vector<size_t> open_loc;            ///< unfilled matrix positions

        Population(){}
        Population(int p){individuals.resize(p);}
        ~Population(){}
        
        /// initialize population of programs. 
        void init(const Individual& starting_model, const Parameters& params);
        
        /// update individual vector size 
        void resize(int pop_size){	individuals.resize(pop_size); }
        
        /// reduce programs to the indices in survivors. 
        void update(vector<size_t> survivors);
        
        /// returns population size
        int size(){ return individuals.size(); }

        /// returns an open location 
        size_t get_open_loc(); 
        
        /// updates open locations to reflect population.
        void update_open_loc();

        /// adds a program to the population. 
        void add(Individual&);
        
        /// setting and getting from individuals vector
        const Individual operator [](size_t i) const {return individuals.at(i);}
        const Individual & operator [](size_t i) {return individuals.at(i);}

        /// return population equations. 
        string print_eqns(bool,string);

        /// return complexity-sorted Pareto front indices. 
        vector<size_t> sorted_front(unsigned);
        
        /// Sort population in increasing complexity.
        struct SortComplexity
        {
            Population& pop;
            SortComplexity(Population& p): pop(p){}
            bool operator()(size_t i, size_t j)
            { 
                return pop.individuals[i].complexity() < pop.individuals[j].complexity();
            }
        };
        /// check for same fitness and complexity to filter uniqueness. 
        struct SameFitComplexity
        {
            Population & pop;
            SameFitComplexity(Population& p): pop(p){}
            bool operator()(size_t i, size_t j)
            {
                return (pop.individuals[i].fitness == pop.individuals[j].fitness &&
                       pop.individuals[i].complexity() == pop.individuals[j].complexity());
            }
        };

    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
 
    bool is_valid_program(vector<std::shared_ptr<Node>>& program, unsigned num_features)
    {
        /*! checks whether program fulfills all its arities. */
        vector<ArrayXd> stack_f; 
        vector<ArrayXb> stack_b;
        MatrixXd X = MatrixXd::Zero(num_features,2); 
        VectorXd y = VectorXd::Zero(2); 
        unsigned i = 0; 
        for (const auto& n : program){
            if ( stack_f.size() >= n->arity['f'] && stack_b.size() >= n->arity['b'])
                n->evaluate(X, y, stack_f, stack_b);
            else{
                std::cout << "Error: ";
                for (const auto& p: program) std::cout << p->name << " ";
                std::cout << "is not a valid program because ";
                std::cout << n->name << " at pos " << i << "is not satisfied\n";
                return false; 
            }
            ++i;
        }
        return true;
    }
   
    void make_tree(vector<std::shared_ptr<Node>>& program, 
                      const vector<std::shared_ptr<Node>>& functions, 
                      const vector<std::shared_ptr<Node>>& terminals, int max_d,  
                      const vector<double>& term_weights, char otype)
    {  
                
        /*!
         * recursively builds a program with complete arguments.
         */
        if (max_d == 0 || r.rnd_flt() < terminals.size()/(terminals.size()+functions.size())) 
        {
            // append terminal 
            vector<size_t> ti, tw;  // indices of valid terminals 
            for (size_t i = 0; i<terminals.size(); ++i)
            {
                if (terminals[i]->otype == otype) // grab terminals matching output type
                {
                    ti.push_back(i);
                    tw.push_back(term_weights[i]);
                }
            }
            auto t = terminals[r.random_choice(ti,tw)];
            //std::cout << t->name << " ";
            program.push_back(t);
        }
        else
        {
            // let fi be indices of functions whose output type matches otype and, if max_d==1,
            // with no boolean inputs (assuming all input data is floating point) 
            vector<size_t> fi;
            for (size_t i = 0; i<functions.size(); ++i)
                if (functions[i]->otype==otype && (max_d>1 || functions[i]->arity['b']==0))
                    fi.push_back(i);
            //std::cout << "fi size: " << fi.size() << "\n";
            if (fi.size()==0){
                std::cout << "---\n";
                std::cout << "f1.size()=0. current program: ";
                for (auto p : program) std::cout << p->name << " ";
                std::cout << "\n";
                std::cout << "otype: " << otype << "\n";
                std::cout << "max_d: " << max_d << "\n";
                std::cout << "functions: ";
                for (auto f: functions) std::cout << f->name << " ";
                std::cout << "\n";
                std::cout << "---\n";
            }
            assert(fi.size() > 0 && "The operator set specified results in incomplete programs.");
            
            // append a random choice from fs            
            auto t = functions[r.random_choice(fi)];
            //std::cout << t->name << " ";
            program.push_back(t);
            
            std::shared_ptr<Node> chosen = program.back();
            // recurse to fulfill the arity of the chosen function
            for (size_t i = 0; i < chosen->arity['f']; ++i)
                make_tree(program, functions, terminals, max_d-1, term_weights,'f');
            for (size_t i = 0; i < chosen->arity['b']; ++i)
                make_tree(program, functions, terminals, max_d-1, term_weights, 'b');

        }

    }

    void make_program(vector<std::shared_ptr<Node>>& program, 
                      const vector<std::shared_ptr<Node>>& functions, 
                      const vector<std::shared_ptr<Node>>& terminals, int max_d, 
                      const vector<double>& term_weights, int dim, char otype)
    {
  
         
        for (unsigned i = 0; i<dim; ++i)    // build trees
            make_tree(program, functions, terminals, max_d, term_weights, otype);
        
        // reverse program so that it is post-fix notation
        std::reverse(program.begin(),program.end());
        assert(is_valid_program(program,terminals.size()));
    }


    void Population::init(const Individual& starting_model, const Parameters& params)
    {
        /*!
         *create random programs in the population, seeded by initial model weights 
         */
        int i = 0;        
        size_t count = -1;
        vector<char> otypes = {'b','f'};
        for (auto& ind : individuals)
        {
            //std::cout << "i: " <<  i << "\n";
            // the first individual is the starting model (i.e., the raw features)
            if (count == -1)
            {
                ind = starting_model;                
                ind.loc = ++count;
                std::cout << ind.get_eqn() + "\n";
                continue;
            }
            // make a program for each individual
            // pick a max depth for this program
            // pick a dimensionality for this individual
            int dim = r.rnd_int(1,params.max_dim);      
            // pick depth from [params.min_depth, params.max_depth]
            int depth =  r.rnd_int(1, params.max_depth);
            
            make_program(ind.program, params.functions, params.terminals, depth,
                         params.term_weights,dim,r.random_choice(params.otypes));

            //std::cout << ind.get_eqn() + "\n";
           
            // set location of individual and increment counter
            ind.loc = ++count;                    
        }
        // define open locations
        update_open_loc(); 
    }
   
   void Population::update(vector<size_t> survivors)
   {

       /*!
        * cull population down to survivor indices.
        */
       vector<size_t> pop_idx(individuals.size());
       std::iota(pop_idx.begin(),pop_idx.end(),0);
       std::reverse(pop_idx.begin(),pop_idx.end());
       for (const auto& i : pop_idx)
           if (!in(survivors,i))
               individuals.erase(individuals.begin()+i);                         
          
       //individuals.erase(std::remove_if(individuals.begin(), individuals.end(), 
       //                  [&survivors](const Individual& ind){ return !in(survivors,ind.loc);}),
       //                  individuals.end());

       // reset the open locations in F matrix 
       update_open_loc();
   
   }

   size_t Population::get_open_loc()
   {
       /*!
        * grabs an open location and removes it from the vector.
        */
       size_t loc = open_loc.back(); open_loc.pop_back();
       return loc;
   }

   void Population::update_open_loc()
   {
       /*!
        * updates open_loc to any locations in [0, 2*popsize-1] not in individuals.loc
        */
      
       vector<size_t> current_locs, new_open_locs;
       
       // get vector of current locations       
       for (const auto& ind : individuals)
           current_locs.push_back(ind.loc);
       
       // find open locations
       size_t i = 0;
       while (i < 2* individuals.size())
       {
           if (!in(current_locs,i))
               new_open_locs.push_back(i);
           ++i;
       }


       // re-assign open locations
       open_loc = new_open_locs;
              
   }
   void Population::add(Individual& ind)
   {
       /*!
        * adds ind to individuals, giving it an open location and bookeeping.
        */

       ind.loc = get_open_loc();
       individuals.push_back(ind);
   }

   string Population::print_eqns(bool just_offspring=false, string sep="\n")
   {
       string output = "";
       int start = 0;
       
       if (just_offspring)
           start = individuals.size()/2;

       for (unsigned int i=start; i< individuals.size(); ++i)
           output += individuals[i].get_eqn() + sep;
       
       return output;
   }

    vector<size_t> Population::sorted_front(unsigned rank=1)
    {
        /* Returns individuals on the Pareto front, sorted by increasign complexity. */
        vector<size_t> pf;
        for (unsigned int i =0; i<individuals.size(); ++i)
        {
            if (individuals[i].rank == rank)
                pf.push_back(i);
        }
        std::sort(pf.begin(),pf.end(),SortComplexity(*this)); 
        auto it = std::unique(pf.begin(),pf.end(),SameFitComplexity(*this));
        pf.resize(std::distance(pf.begin(),it));
        return pf;
    }
    
}//FT    
#endif
