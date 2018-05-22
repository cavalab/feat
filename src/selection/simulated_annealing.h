/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef SIMANNEAL_H
#define SIMANNEAL_H

namespace FT{
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class SimAnneal
     */
    struct SimAnneal : SelectionOperator
    {
        /** SimAnneal based selection and survival methods. */

        SimAnneal(bool surv){ name = "simanneal"; survival = surv; t0 = 10;};
        
        ~SimAnneal(){}
       
        vector<size_t> select(Population& pop, const MatrixXd& F, const Parameters& params);
        vector<size_t> survive(Population& pop, const MatrixXd& F, const Parameters& params);
    private:
        double t;           ///< annealing temperature
        double t0;          ///< initial temperature
    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions
    vector<size_t> SimAnneal::select(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Selects parents for making offspring.  
         *
         * @params pop: population of programs, all potential parents
         * * @params F: n_samples x 2 * popsize matrix of program behaviors. 
         * @params params: parameters.
         *
         * @returns selected: vector of indices corresponding to offspring that are selected.
         *      
         */
      
        int P = pop.size(); // index P is where the offspring begin, and also the size of the pop
       
        vector<size_t> all_idx(pop.size());
        std::iota(all_idx.begin(), all_idx.end(), 0);
        return all_idx;
    }

    vector<size_t> SimAnneal::survive(Population& pop, const MatrixXd& F, const Parameters& params)
    {
        /* Selects the offspring for survival using simulated annealing.
         *
         * Offspring are compared to their parents. The probability of an offspring, R, replacing 
         * its parent, S, is given by
         * 
         *      P(t, R, S ) = exp( ( fitness(S) - fitness(R) ) / t)
         *
         * where t is the temperature. 
         *
         * @params pop: population of programs, parents + offspring.
         * @params F: n_samples x 2 * popsize matrix of program behaviors. 
         * @params params: parameters.
         *
         * @returns selected: vector of indices corresponding to offspring that are selected.
         *      
         */
     
        // decrease temperature linearly based on current generation
        // cooling schedule: Tg = (0.9)^g * t0, g = current generation
        this->t = pow(0.9, double(params.current_gen))*this->t0;  
        /* cout << "t: " << this->t << "\n"; */

        int P = F.cols()/2; // index P is where the offspring begin, and also the size of the pop
        vector<size_t> selected(P); 
        #pragma omp parallel for        
        for (unsigned i = P; i < F.cols(); ++i)
        {
            Individual& offspring = pop.individuals.at(i);
            /* cout << "offspring: " << offspring.get_eqn() << "\n"; */
            int pid = offspring.parent_id.at(0);
            bool found = false;
            int j = 0;
            while (!found && j < P)
            {
                if ( pop.individuals.at(j).id == pid)
                    found=true;
                else
                    ++j;
            }
            Individual& parent = pop.individuals.at(j);
            /* cout << "parent: " << parent.get_eqn() << "\n"; */
            /* cout << "offspring fitness: " << offspring.fitness << "\n"; */
            /* cout << "parent fitness: " << parent.fitness << "\n"; */           
            double probability = exp ( (parent.fitness - offspring.fitness)/this->t );

            /* cout << "probability: " << probability << "\n"; */
            if (r() < probability)
                selected.at(i-P) = i;
            else
                selected.at(i-P) = j;
        }
        return selected;
    }
}
#endif
