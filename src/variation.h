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
            /*!
             * @brief constructor
             */
            Variation(float cr): cross_rate(cr) {}
            
            /*!
             * @brief destructor
             */
            ~Variation(){}

            /*!
             * @brief method to handle variation of population
             */
            void vary(Population& pop, const vector<size_t> parents, const Parameters& params);
            
        private:
            /*!
             * @brief crossover 
             */
            bool cross(const Individual& mom, const Individual& dad, Individual& child,
                       const Parameters& params);

            /*!
             * @brief mutation
             */
            bool mutate(const Individual& mom, Individual& child, const Parameters& params);
        
            /*!
             * @brief splice two programs together
             */
            vector<Node> splice_programs(const vector<Node>& v1, size_t i1, size_t j1, 
                                                 const vector<Node>& v2, size_t i2, size_t j2);
       
            float cross_rate;     ///< fraction of crossover in total variation
    };


    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Variation::vary(Population& pop, const vector<size_t> parents, const Parameters& params)
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
        
        bool pass;                      // pass check for children undergoing variation       
        while (pop.size() < 2*params.pop_size)
        {
            Individual child;           // new individual

            if ( r() < cross_rate)      // crossover
            {
                // get random mom and dad 
                int mom = r.random_choice(parents);
                int dad = r.random_choice(parents);
                // create child
                pass = cross(pop[mom],pop[dad],child,params);
                
            }
            else                        // mutation
            {
                // get random mom
                int mom = r.random_choice(parents);
                // create child
                pass = mutate(pop[mom],child,params);
            }
            if (pass)                   // congrats! you produced a viable child.
            {
                // give child an open location in F
                child.loc = pop.get_open_loc(); 
                //push child into pop
                pop.individuals.push_back(child);
            }
        }
    }

    bool Variation::mutate(const Individual& mom, Individual& child, const Parameters& params)
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
            if (r() < 1/n)  // mutate p. WIP: change '1' to node weighted probability
            {
                vector<Node> replacements;  // potential replacements for p

                if (p.total_arity() > 0) // then it is an instruction
                {
                    // find instructions with matching in/out types and arities
                    for (const auto& f: params.functions)
                    {
                        if (f.otype == p.otype && f.arity_f==p.arity_f && f.arity_b==p.arity_b)
                            replacements.push_back(f);
                    }
                }
                else                    // otherwise it is a terminal
                {
                    // WIP: add terminal weights here
                    // find terminals with matching output types
                    vector<Node> replacements;
                    for (const auto& t : params.terminals)
                    {
                        if (t.otype == p.otype)
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

    bool Variation::cross(const Individual& mom, const Individual& dad, Individual& child, 
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
           if (not_in(otypes,p.otype)) otypes.push_back(p.otype);
       for (const auto& p : dad.program)
           if (not_in(otypes,p.otype)) otypes.push_back(p.otype); 

       // get valid subtree locations
       vector<size_t> locs;
       for (size_t i =0; i<mom.size(); ++i) if (in(otypes,mom[i].otype)) locs.push_back(i);
       //std::iota(locs.begin(),locs.end());

       // get subtree
       size_t j1 = r.random_choice(locs);
       size_t i1 = mom.subtree(j1);


       // get locations in dad's program that match the subtree type picked from mom
       vector<size_t> dlocs;
       for (size_t i =0; i<dad.size(); ++i) if (dad[i].otype == mom[j1].otype) dlocs.push_back(i);
       //std::iota(dlocs.begin(),dlocs.end());
       
       // get dad subtree
       size_t j2 = r.random_choice(dlocs);
       size_t i2 = dad.subtree(j2);

       // make child program by splicing mom and dad
       child.program = splice_programs(mom.program, i1, j1, dad.program, i2, j2);
        
       // check child depth and dimensionality
       return child.size() <= params.max_size && child.get_dim() <= params.max_dim;
    }
    
    // swap vector subsets with different sizes. 
    vector<Node> Variation::splice_programs(const vector<Node>& v1, size_t i1, size_t j1, 
                                         const vector<Node>& v2, size_t i2, size_t j2)
    {
        /*!
         * @brief swap vector subsets with different sizes. 
         * constructs a vector made of v1[0:i1], v2[i2:j2], v1[i1:end].
         *
         * Input:
         *
         *      v1: root parent 
         *          i1: start of splicing 
         *          j1: end of root splicing
         *      v2: donating parent
         *          i2: start of donation
         *          j2: end of donation
         *
         * Output: 
         *
         *      vnew: new vector 
         */

        // size difference between subtrees  
        vector<Node> vnew;
        vnew.insert(vnew.end(),v1.begin(),v1.begin()+i1+1);     // beginning of v1
        vnew.insert(vnew.end(),v2.begin()+i2,v2.begin()+j2+1);  // spliced in v2 portion
        vnew.insert(vnew.end(),v1.begin()+j1+1,v1.end());       // end of v1
    }
}
#endif
