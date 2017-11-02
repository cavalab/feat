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
            void point_mutate(Individual& child, const Parameters& params);
            void insert_mutate(Individual& child, const Parameters& params);
            void delete_mutate(Individual& child, const Parameters& params);
 
            /// splice two programs together
            vector<std::shared_ptr<Node>> splice_programs(vector<std::shared_ptr<Node>>& v1, 
                                                          size_t i1, size_t j1, 
                                                          vector<std::shared_ptr<Node>>& v2,
                                                          size_t i2, size_t j2);
            /// debugging printout of crossover operation.
            void print_cross(Individual&,size_t,size_t,Individual&, size_t, size_t, Individual&,
                             bool after=true);       
            
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
            
                params.msg("crossing " + pop.individuals[mom].get_eqn() + " with " + 
                       pop.individuals[dad].get_eqn() + " produced " + child.get_eqn() + ", pass: " 
                       + std::to_string(pass),2);    
            }
            else                        // mutation
            {
                // get random mom
                int mom = r.random_choice(parents);                
                params.msg("mutating " + pop.individuals[mom].get_eqn(), 2);
                // create child
                pass = mutate(pop.individuals[mom],child,params);
                
                params.msg("mutating " + pop.individuals[mom].get_eqn() + " produced " + 
                        child.get_eqn() + ", pass: " + std::to_string(pass),2);
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

    bool Variation::mutate(Individual& mom, Individual& child, const Parameters& params)
    {
        /*!
         * chooses uniformly between point mutation, insert mutation and delete mutation 
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
        
        float rf = r();
        if (rf < 1.0/3.0 && child.get_dim() > 1)
           delete_mutate(child,params); 
        else if (rf < 2.0/3.0 && child.size() < params.max_size)
            insert_mutate(child,params);
        else
            point_mutate(child,params);
        assert(is_valid_program(child.program)); 
        // check child depth and dimensionality
        return child.size() <= params.max_size && child.get_dim() <= params.max_dim;
    }

    void Variation::point_mutate(Individual& child, const Parameters& params)
    {
        std::cout << "point mutation\n";
        /* 1/n point mutation. */
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

    }
    void Variation::insert_mutate(Individual& child, const Parameters& params)
    {
        std::cout << "insert mutation\n";
        /* 1/n point mutation. */
        float n = child.size(); 
        if (r()<0.5 || child.get_dim() == params.max_dim)
        {
            // loop thru child's program
            for (unsigned i = 0; i< child.program.size(); ++i)
            {
                if (r() < 1/n)  // mutate p. TODO: change '1' to node weighted probability
                {
                    params.msg("insert mutating node " + child.program[i]->name, 2);
                    vector<std::shared_ptr<Node>> insertion;  // inserted segment
                    vector<std::shared_ptr<Node>> fns;  // potential fns 
                    
                    // find instructions with matching in/out types and arities
                    for (const auto& f: params.functions)
                    {
                        if (f->arity[child.program[i]->otype] > 0)
                            fns.push_back(f);                        
                    }
                    make_program(insertion, fns, params.terminals, 1, child.program[i]->otype, 
                                 params.term_weights,1);
                    
                    for (auto& ins : insertion){    // replace first argument in insertion
                        if (ins->otype == child.program[i]->otype 
                                && ins->arity['f']==child.program[i]->arity['f'] 
                                && ins->arity['b']==child.program[i]->arity['b'])
                        {
                            ins = child.program[i];
                            continue;
                        }

                    }
                    child.program.erase(child.program.begin()+i);
                    child.program.insert(child.program.begin()+i, insertion.begin(), 
                                         insertion.end());
                    i += insertion.size()-1;
               }
                
            }
        }
        else    // add a dimension
        {
            vector<std::shared_ptr<Node>> insertion; // new dimension
            make_program(insertion, params.functions, params.terminals, 1, params.otype, 
                         params.term_weights,1);
            child.program.insert(child.program.end(),insertion.begin(),insertion.end());
        }
    }

    void Variation::delete_mutate(Individual& child, const Parameters& params)
    {
        std::cout << "deletion mutation\n";
        /* 1/n deletion mutation. deletes dimensions. */
        std::cout << "program: " + child.program_str() + "\n";
        vector<size_t> roots = child.roots();
        std::cout << "# roots: " << roots.size() << "\n";
        size_t end = r.random_choice(roots); // TODO: weight with node probabilities
        std::cout << "root chosen: " << end << "\n";
        size_t start = child.subtree(end);  
        if (params.verbosity >=2)
        { 
            std::string s="";
            for (unsigned i = start; i<end; ++i) s+= child.program[i]->name + " ";
            params.msg("deleting " + s, 2);
        }    
        child.program.erase(child.program.begin()+start,child.program.begin()+end+1);
        std::cout << "result of delete mutation: " + child.program_str() + "\n"; 
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
                    
        bool subtree = r() <0.5;     // half the time, do subtree xo. 
                                     // half the time, swap dimensions.
        vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
        size_t i1, j1, i2, j2;       // i1-j1: mom portion, i2-j2: dad portion
        
        if (subtree) 
        {
            std::cout << "subtree xo\n";
            // limit xo choices to matching output types in the programs. 
            vector<char> otypes;
            for (const auto& p : mom.program)
                otypes.push_back(p->otype);
            for (const auto& p : dad.program)
                if (!in(otypes,p->otype))    // if dad doesn't have this otype, remove it 
                    otypes.erase(std::remove(otypes.begin(),otypes.end(),p->otype),otypes.end()); 

            // get valid subtree locations
            for (size_t i =0; i<mom.size(); ++i) 
                if (in(otypes,mom[i]->otype)) 
                    mlocs.push_back(i);       
            
            j1 = r.random_choice(mlocs);    

            // get locations in dad's program that match the subtree type picked from mom
            for (size_t i =0; i<dad.size(); ++i) 
                if (dad[i]->otype == mom[j1]->otype) dlocs.push_back(i);
        } 
        else             // half the time, pick a root node
        {
            std::cout << "root xo\n";
            mlocs = mom.roots();
            dlocs = dad.roots();
            std::cout << "random choice mlocs\n";
            j1 = r.random_choice(mlocs);    
        }
        // get subtree        
        std::cout << "get subtree (j1 = " << j1 << ")";
        std::cout << "of program " + mom.program_str() + "\n";
        i1 = mom.subtree(j1);
                             
        // get dad subtree
        j2 = r.random_choice(dlocs);
        i2 = dad.subtree(j2);
        std::cout << "i1: " << i1 << ", j1: " << j1 << ", i2: " << i2 << ", j2: " << j2 << "\n";

        if (params.verbosity >= 2) 
            print_cross(mom,i1,j1,dad,i2,j2,child, false);
        
        // make child program by splicing mom and dad
        child.program = splice_programs(mom.program, i1, j1, dad.program, i2, j2);
                     
        if (params.verbosity >= 2) 
            print_cross(mom,i1,j1,dad,i2,j2,child);     

        assert(is_valid_program(child.program));
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
                                size_t j2, Individual& child, bool after)
    {
        std::cout << "attempting the following crossover:\n";
        for (int i =0; i<mom.program.size(); ++i){
           if (i>= i1 && i<= j1) 
               std::cout << "_";
           std::cout << mom.program[i]->name << " ";
        }
        std::cout << "\n";
       
        for (int i =0; i<dad.program.size(); ++i){
            if (i>= i2 && i<= j2) 
                std::cout << "_";
            std::cout << dad.program[i]->name << " ";
        }
        std::cout << "\n";
        if (after)
        {
            std::cout << "child after cross:\n";
            for (auto& p : child.program)
                std::cout << p->name << " "; 
            std::cout << "\n";
        }
    }
}
#endif
