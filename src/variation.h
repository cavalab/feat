/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

namespace FT{
    struct Individual;  // forward declarations
    struct Parameters;
    struct Population;
    //Rnd r;
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Variation
     */ 
    class Variation 
    {
        /*!
         * methods for crossover and mutation of programs.
         */

        public:
        
            /// constructor
            Variation(float cr): cross_rate(cr) {}
                       
            /// update cross rate
            void set_cross_rate(float cr)
            {
            	cross_rate = cr;
            }
            
            /// return current cross rate
            float get_cross_rate()
            {
            	return cross_rate;
            }
            
             /// destructor
            ~Variation(){}

            /// method to handle variation of population
            void vary(Population& pop, const vector<size_t>& parents, const Parameters& params);
            
        private:
        
            /// crossover
            bool cross(Individual& mom, Individual& dad, Individual& child,
                       const Parameters& params);

            /// mutation
            bool mutate(Individual& mom, Individual& child, const Parameters& params);
        
            /// splice two programs together
            vector<std::shared_ptr<Node>> splice_programs(vector<std::shared_ptr<Node>>& v1, 
                                                          size_t i1, size_t j1, 
                                                          vector<std::shared_ptr<Node>>& v2,
                                                          size_t i2, size_t j2);
            /// debugging printout of crossover operation.
            void print_cross(Individual&,size_t,size_t,Individual&, size_t, size_t, Individual&);       
            
            float cross_rate;     ///< fraction of crossover in total variation
    };


    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Variation::vary(Population& pop, const vector<size_t>& parents, const Parameters& params)
    {
        /*!
         * performs variation on the current population. 
         *
         * Input:
         *
         *      	pop: current population
         *      	parents: indices of population to use for variation
         *      	params: fewtwo parameters
         *
         * Output:
         *
         *      	pop: appends params.pop_size offspring derived from parent variation
         */
        // vector<size_t> other_parents;    // parents other than one chosen for crossover
        bool pass;                      // pass check for children undergoing variation       
        while (pop.size() < 2*params.pop_size)
        {
            Individual child;           // new individual

            if ( r() < cross_rate)      // crossover
            {
                // get random mom and dad 
                int mom = r.random_choice(parents);
          //      other_parents.assign(parents.begin(),parents.begin()+mom);
            //    other_parents.insert(other_parents.end();parents.begin()+mom+1,parents.end());
                int dad = r.random_choice(parents);
                // create child
                params.msg("crossing " + pop.individuals[mom].get_eqn() + " with " + 
                           pop.individuals[dad].get_eqn(), 2);
                pass = cross(pop.individuals[mom],pop.individuals[dad],child,params);
                
            }
            else                        // mutation
            {
                // get random mom
                int mom = r.random_choice(parents);                
                params.msg("mutating " + pop.individuals[mom].get_eqn(), 2);
                // create child
                pass = mutate(pop.individuals[mom],child,params);
            }
            
            params.msg("child: " + child.get_eqn() + ", pass: " + std::to_string(pass),2);
            
            if (pass)                   // congrats! you produced a viable child.
            {
                // give child an open location in F
                child.loc = pop.get_open_loc(); 
                //push child into pop
                pop.individuals.push_back(child);
            }
        }
    }

    bool Variation::mutate(Individual& mom, Individual& child, const Parameters& params)
    {
        /*!
         * 1/n point mutation
         * 
         * Input:
         *
         *      	mom: root parent
         *
         * Output:
         *
         *      	child: copy of mom with some mutations
         */    

        // make child a copy of mom
        child = mom; 
        float n = child.size();

        // loop thru child's program
        for (auto& p : child.program)
        {

            if (r() < 1/n)  // mutate p. TODO: change '1' to node weighted probability
            {
                params.msg("mutating node " + p->name, 2);
                vector<std::shared_ptr<Node>> replacements;  // potential replacements for p

                if (p->total_arity() > 0) // then it is an instruction
                {
                    // find instructions with matching in/out types and arities
                    for (const auto& f: params.functions)
                    {
                        if (f->otype == p->otype && f->arity['f']==p->arity['f'] && 
                                f->arity['b']==p->arity['b'])
                            replacements.push_back(f);
                    }
                }
                else                    // otherwise it is a terminal
                {
                    // TODO: add terminal weights here
                    // find terminals with matching output types
                                      
                    for (const auto& t : params.terminals)
                    {
                        if (t->otype == p->otype)
                            replacements.push_back(t);
                    }                                       
                }
                // replace p with a random one
                p = r.random_choice(replacements);  
            }
        }
        // check child depth and dimensionality
        return child.size() <= params.max_size && child.get_dim() <= params.max_dim;
    }

    bool Variation::cross(Individual& mom, Individual& dad, Individual& child, 
                          const Parameters& params)
    {
        /*!
         * subtree crossover
         *
         * Input:
         *
         *      	mom: root parent
         *      	dad: parent from which subtree is chosen
         *
         * Output:
         *
         *      	child: mom with dad subtree graft
         */
                    
       // we must limit ourselves to matching output types in the programs. 
       vector<char> otypes;
       for (const auto& p : mom.program)
           if (not_in(otypes,p->otype)) otypes.push_back(p->otype);
       for (const auto& p : dad.program)
           if (not_in(otypes,p->otype)) otypes.push_back(p->otype); 

       // get valid subtree locations
       vector<size_t> locs;
       for (size_t i =0; i<mom.size(); ++i) 
           if (in(otypes,mom[i]->otype)) locs.push_back(i);       

       // get subtree
       size_t j1 = r.random_choice(locs);
       size_t i1 = mom.subtree(j1);
   
       // get locations in dad's program that match the subtree type picked from mom
       vector<size_t> dlocs;
       for (size_t i =0; i<dad.size(); ++i) 
           if (dad[i]->otype == mom[j1]->otype) dlocs.push_back(i);
                     
       // get dad subtree
       size_t j2 = r.random_choice(dlocs);
       size_t i2 = dad.subtree(j2);
       
       
       // make child program by splicing mom and dad
       child.program = splice_programs(mom.program, i1, j1, dad.program, i2, j2);
                    
       if (params.verbosity >= 2) 
           print_cross(mom,i1,j1,dad,i2,j2,child);     
      
       // check child depth and dimensionality
       return child.size() <= params.max_size && child.get_dim() <= params.max_dim;
    }
    
    // swap vector subsets with different sizes. 
    vector<std::shared_ptr<Node>> Variation::splice_programs(
                                     vector<std::shared_ptr<Node>>& v1, size_t i1, size_t j1, 
                                     vector<std::shared_ptr<Node>>& v2, size_t i2, size_t j2)
    {
        /*!
         * swap vector subsets with different sizes. 
         * constructs a vector made of v1[0:i1], v2[i2:j2], v1[i1:end].
         *
         * Input:
         *
         *      v1: root parent 
         *          i1: start of splicing segment 
         *          j1: end of splicing segment
         *      v2: donating parent
         *          i2: start of donation
         *          j2: end of donation
         *
         * Output: 
         *
         *      vnew: new vector 
         */

        // size difference between subtrees  
        vector<std::shared_ptr<Node>> vnew;
        vnew.insert(vnew.end(),v1.begin(),v1.begin()+i1);     // beginning of v1
        vnew.insert(vnew.end(),v2.begin()+i2,v2.begin()+j2+1);  // spliced in v2 portion
        vnew.insert(vnew.end(),v1.begin()+j1+1,v1.end());       // end of v1
        return vnew;
    }
    
    void Variation::print_cross(Individual& mom, size_t i1, size_t j1, Individual& dad, size_t i2, 
                                size_t j2, Individual& child)
    {
        std::cout << "attempting the following crossover:\n";
           for (int i =0; i<mom.program.size(); ++i){
               if (i>= i1 && i<= j1) 
                   std::cout << "*";
               std::cout << mom.program[i]->name << " ";
           }
           std::cout << "\n";
           
           for (int i =0; i<dad.program.size(); ++i){
               if (i>= i2 && i<= j2) 
                   std::cout << "*";
               std::cout << dad.program[i]->name << " ";
           }
           std::cout << "\n";
           std::cout << "child after cross:\n";
           for (auto& p : child.program)
                std::cout << p->name << " "; 
           std::cout << "\n";
    }
}
#endif
