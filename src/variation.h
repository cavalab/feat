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
            bool pass=false;                      // pass check for children undergoing variation     
   
            while (!pass)
            {
                Individual child;           // new individual

                if ( r() < cross_rate)      // crossover
                {
                    // get random mom and dad 
                    int mom = r.random_choice(parents);
                    int dad = r.random_choice(parents);
                    // create child
                    params.msg("crossing " + pop.individuals[mom].get_eqn() + " with " + 
                               pop.individuals[dad].get_eqn(), 2);
                    pass = cross(pop.individuals[mom],pop.individuals[dad],child,params);
                
                    params.msg("crossing " + pop.individuals[mom].get_eqn() + " with " + 
                           pop.individuals[dad].get_eqn() + " produced " + child.get_eqn() + 
                           ", pass: " + std::to_string(pass),2);    
                }
                else                        // mutation
                {
                    // get random mom
                    int mom = r.random_choice(parents);                
                    params.msg("mutating " + pop.individuals[mom].get_eqn() + "(" + 
                            pop.individuals[mom].program_str() + ")", 2);
                    // create child
                    pass = mutate(pop.individuals[mom],child,params);
                    
                    params.msg("mutating " + pop.individuals[mom].get_eqn() + " produced " + 
                            child.get_eqn() + ", pass: " + std::to_string(pass),2);
                }
                if (pass)
                {
                    assert(child.size()>0);
                    assert(pop.open_loc.size()>i-start);
                    params.msg("assigning " + child.program_str() + " to pop.individuals[" + 
                        std::to_string(i) + "] with pop.open_loc[" + std::to_string(i-start) + 
                        "]=" + std::to_string(pop.open_loc[i-start]),2);

                    pop.individuals[i] = child;
                    pop.individuals[i].loc = pop.open_loc[i-start];                   
                }
            }    
       }
      
       pop.update_open_loc();
    }

    bool Variation::mutate(Individual& mom, Individual& child, const Parameters& params)
    {
        /*!
         * chooses uniformly between point mutation, insert mutation and delete mutation 
         * 
         * @param   mom: parent
         * @param   child: offspring produced by mutating mom 
         * @param   params: parameters
         * 
         * @return  true if valid child, false if not 
         */    

        // make child a copy of mom
        child.program = mom.program; 
        child.p = mom.p; 

        float rf = r();
        if (rf < 1.0/3.0 && child.get_dim() > 1){
            delete_mutate(child,params); 
            assert(is_valid_program(child.program, params.num_features));
        }
        else if (rf < 2.0/3.0 && child.size() < params.max_size)
        {
            insert_mutate(child,params);
            assert(is_valid_program(child.program, params.num_features));
        }
        else
        {        
            point_mutate(child,params);
            assert(is_valid_program(child.program, params.num_features));
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
         * @returns modified child
         * */
        params.msg("\tpoint mutation",2);
        float n = child.size(); 
        unsigned i = 0;
        // loop thru child's program
        for (auto& p : child.program)
        {
            if (r() < child.get_p(i)/n)  // mutate p. 
            {
                params.msg("\t\tmutating node " + p->name, 2);
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
            ++i; 
        }

    }
    void Variation::insert_mutate(Individual& child, const Parameters& params)
    {        
        /*! insertion mutation. 
         * @param child: indiviudal to be mutated
         * @param params: parameters
         * @returns modified child
         * */
        
        params.msg("\tinsert mutation",2);
        float n = child.size(); 
        
        if (r()<0.5 || child.get_dim() == params.max_dim)
        {
            // loop thru child's program
            for (unsigned i = 0; i< child.program.size(); ++i)
            {
                
                if (r() < child.get_p(i)/n)  // mutate with weighted probability
                {
                    params.msg("\t\tinsert mutating node " + child.program[i]->name, 2);
                    vector<std::shared_ptr<Node>> insertion;  // inserted segment
                    vector<std::shared_ptr<Node>> fns;  // potential fns 
                    
                    // find instructions with matching output types and a matching arity to i
                    for (const auto& f: params.functions)
                    { 
                        // find fns with matching output types that take this node type as arg
                        if (f->arity[child.program[i]->otype] > 0 && 
                                f->otype==child.program[i]->otype )
                        { 
                            // make sure there are satisfactory types in terminals to fill fns' 
                            // args
                            if (child.program[i]->otype=='b')
                                if (in(params.dtypes,'b') || f->arity['b']==1)
                                    fns.push_back(f);
                            else if (child.program[i]->otype=='f')
                                if (f->arity['b']==0 || in(params.dtypes,'b') )
                                    fns.push_back(f);              
                        }
                    }

                    if (fns.size()==0)  // if no insertion functions match, skip
                        continue;

                    // choose a function to insert                    
                    insertion.push_back(r.random_choice(fns));
                    
                    unsigned fa = insertion.back()->arity['f']; // float arity
                    unsigned ba = insertion.back()->arity['b']; // bool arity
                    // decrement function arity by one for child node 
                    if (child.program[i]->otype=='f') --fa;
                    else --ba; 
                    // push back new arguments for the rest of the function
                    for (unsigned j = 0; j< fa; ++j)
                        make_tree(insertion,params.functions,params.terminals,0,
                                  params.term_weights,'f');
                    if (child.program[i]->otype=='f')    // add the child now if float
                        insertion.push_back(child.program[i]);
                    for (unsigned j = 0; j< ba; ++j)
                        make_tree(insertion,params.functions,params.terminals,0,
                                  params.term_weights,'b');
                    if (child.program[i]->otype=='b') // add the child now if bool
                        insertion.push_back(child.program[i]);
                    // post-fix notation
                    std::reverse(insertion.begin(),insertion.end());
                    
                    // put the new code in the program 
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
            make_program(insertion, params.functions, params.terminals, 1,  
                         params.term_weights,1,r.random_choice(params.otypes));
            child.program.insert(child.program.end(),insertion.begin(),insertion.end());
        }
    }

    void Variation::delete_mutate(Individual& child, const Parameters& params)
    {

        /*! deletion mutation. works by pruning a dimension. 
         * @param child: individual to be mutated
         * @param params: parameters  
         * @return mutated child
         * */
        params.msg("\tdeletion mutation",2);
        params.msg("\t\tprogram: " + child.program_str(),2);
        vector<size_t> roots = child.roots();
        
        size_t end = r.random_choice(roots,child.p); 
        size_t start = child.subtree(end);  
        if (params.verbosity >=2)
        { 
            std::string s="";
            for (unsigned i = start; i<end; ++i) s+= child.program[i]->name + " ";
            params.msg("\t\tdeleting " + std::to_string(start) + " to " + std::to_string(end) 
                       + ": " + s, 2);
        }    
        child.program.erase(child.program.begin()+start,child.program.begin()+end+1);
        params.msg("\t\tresult of delete mutation: " + child.program_str(), 2);
    }

    bool Variation::cross(Individual& mom, Individual& dad, Individual& child, 
                          const Parameters& params)
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
                    
        bool subtree = r() <0.5;     // half the time, do subtree xo. 
                                     // half the time, swap dimensions.
        vector<size_t> mlocs, dlocs; // mom and dad locations for consideration
        size_t i1, j1, i2, j2;       // i1-j1: mom portion, i2-j2: dad portion
        
        if (subtree) 
        {
            params.msg("\tsubtree xo",2);
            // limit xo choices to matching output types in the programs. 
            vector<char> d_otypes;
            for (const auto& p : dad.program)
                if(!in(d_otypes,p->otype))
                    d_otypes.push_back(p->otype);
            
            // get valid subtree locations
            for (size_t i =0; i<mom.size(); ++i) 
                if (in(d_otypes,mom[i]->otype)) 
                    mlocs.push_back(i);       
            if (mlocs.size()==0)        // mom and dad have no overlapping types, can't cross
            {
                std::cout << "\tno overlapping types between " + mom.program_str() + "," 
                             + dad.program_str() + "\n";
                return 0;               
            }
            j1 = r.random_choice(mlocs,mom.get_p(mlocs));    

            // get locations in dad's program that match the subtree type picked from mom
            for (size_t i =0; i<dad.size(); ++i) 
                if (dad[i]->otype == mom[j1]->otype) dlocs.push_back(i);
        } 
        else             // half the time, pick a root node
        {
            params.msg("\troot xo",2);
            mlocs = mom.roots();
            dlocs = dad.roots();
            params.msg("\t\trandom choice mlocs (size "+std::to_string(mlocs.size())+")",2);
            j1 = r.random_choice(mlocs,mom.get_p(mlocs));   // weighted probability choice    
        }
        // get subtree              
        i1 = mom.subtree(j1);
                             
        // get dad subtree
        j2 = r.random_choice(dlocs);
        i2 = dad.subtree(j2); 
               
        // make child program by splicing mom and dad
        child.program = splice_programs(mom.program, i1, j1, dad.program, i2, j2);
                     
        if (params.verbosity >= 2) 
            print_cross(mom,i1,j1,dad,i2,j2,child);     

        assert(is_valid_program(child.program,params.num_features));
        // check child depth and dimensionality
        return child.size()>0 && child.size() <= params.max_size 
                    && child.get_dim() <= params.max_dim;
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
         * @param   v1: root parent 
         * @param       i1: start of splicing segment 
         * @param       j1: end of splicing segment
         * @param   v2: donating parent
         * @param       i2: start of donation
         * @param       j2: end of donation
         *
         * @return  vnew: new vector 
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
}
#endif
