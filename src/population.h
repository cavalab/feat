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
        
        /*!
         * @brief initialize population of programs. 
         */
        void init(const Parameters& params);
        
        /*!
         * @brief update individual vector size 
         */
        void resize(int &pop_size)
        {
        	individuals.resize(pop_size);
        }
        
        /*!
         * @brief reduce programs to the indices in survivors.
         */
        void update(vector<size_t> survivors);
        
        /*!
         * @brief returns population size
         */
        int size(){return individuals.size();}

        /*!
         * @brief returns an open location 
         */
        size_t get_open_loc(); 
        
        /*!
         * @brief updates open locations to reflect population.
         */
        void update_open_loc();

        /*!
         * @brief adds a program to the population. 
         */
        void add(Individual&);
        
        /*!
         * @brief setting and getting from individuals vector
         */
        const Individual operator [](size_t i) const {return individuals.at(i);}
        const Individual & operator [](size_t i) {return individuals.at(i);}

        // make a program.
        //void make_program(vector<Node>& program, const vector<Node>& functions, 
        //                          const vector<Node>& terminals, int max_d, char otype, 
        //                          const vector<double>& term_weights);

    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void make_program(vector<std::shared_ptr<Node>>& program, const vector<std::shared_ptr<Node>>& functions, 
                                  const vector<std::shared_ptr<Node>>& terminals, int max_d, char otype, 
                                  const vector<double>& term_weights)
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
            program.push_back(terminals[r.random_choice(ti,tw)]);
        }
        else
        {
            // let fs be an index of functions whose output type matches ntype and with an input   
            // type of float if max_d > 1 (assuming all input data is continouous) 
            vector<size_t> fi;
            for (size_t i = 0; i<functions.size(); ++i)
            {
                if (functions[i]->otype == otype && (max_d>1 || functions[i]->arity['b']==0))
                    fi.push_back(i);
            }
            // append a random choice from fs            
            program.push_back(functions[r.random_choice(fi)]);
            
            std::shared_ptr<Node> chosen = program.back();
            // recurse to fulfill the arity of the chosen function
            for (size_t i = 0; i < chosen->arity['f']; ++i)
                make_program(program, functions, terminals, max_d-1, 'f', term_weights);
            for (size_t i = 0; i < chosen->arity['b']; ++i)
                make_program(program, functions, terminals, max_d-1, 'b', term_weights);

        }
    }

    

    void Population::init(const Parameters& params)
    {
        /*!
         *create random programs in the population, seeded by initial model weights 
         */
        std::cout << "population size: " << individuals.size() << "\n";
        size_t count = 0;
        for (auto& ind : individuals){
            // make a program for each individual
            // pick a max depth for this program
            // pick a dimensionality for this individual
            int dim = r.rnd_int(1,params.max_dim);

            for (unsigned int i = 0; i<dim; ++i)
            {
                // pick depth from [params.min_depth, params.max_depth]
                int depth =  r.rnd_int(1, params.max_depth);
                
                make_program(ind.program, params.functions, params.terminals, depth,
                             params.otype, params.term_weights);               
                
            }
            // reverse program so that it is post-fix notation
            std::reverse(ind.program.begin(),ind.program.end());
            std::cout << ind.get_eqn(params.otype) << "\n"; // test output
            
            // set location of individual and increment counter
            ind.loc = count;         
            ++count;               
        }
        // define open locations
        update_open_loc(); 
    }
   
   void Population::update(vector<size_t> survivors)
   {
       /*!
        * cull population down to survivor indices.
        */
       
      individuals.erase(std::remove_if(individuals.begin(), individuals.end(), 
                        [&survivors](const Individual& ind){ return not_in(survivors,ind.loc);}),
                        individuals.end());

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
           if (not_in(current_locs,i))
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


}
#endif
