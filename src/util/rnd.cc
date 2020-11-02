/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "rnd.h"

namespace FT {

    namespace Util{
    
        Rnd* Rnd::instance = NULL;
        
        Rnd::Rnd()
        {
            /*!
             * need a random generator for each core to do multiprocessing
             */
            //cout << "Max threads are " <<omp_get_max_threads()<<"\n";
            rg.resize(omp_get_max_threads());                      
        }

        Rnd* Rnd::initRand()
        {
            if (!instance)
            {
                instance = new Rnd();
            }

            return instance;
        }

        void Rnd::destroy()
        {
            if (instance)
                delete instance;
                
            instance = NULL;
        }
       
        void Rnd::set_seed(int new_seed)
        { 
            /*!
             * set seeds for each core's random number generator
             */
            if (new_seed == -1)
            /* if seed is -1, choose a random seed. */
            {

                int imax = std::numeric_limits<int>::max();
                
                std::uniform_int_distribution<> dist(0, imax);

                this->seed = dist(rg.at(0));

                this->set_seed(this->seed);
            }
            else     
            /* seed first rg with seed, then seed rest with random ints 
             * from rg[0].
             */
            {
                this->seed = new_seed;

                rg.at(0).seed(this->seed);
                
                int imax = std::numeric_limits<int>::max();
                
                std::uniform_int_distribution<> dist(0, imax);

                for (size_t i = 1; i < rg.size(); ++i)
                    rg.at(i).seed(dist(rg.at(0)));                     
            }
        }


        int Rnd::rnd_int( int lowerLimit, int upperLimit ) 
        {
            std::uniform_int_distribution<> dist( lowerLimit, upperLimit );
            return dist(rg.at(omp_get_thread_num()));
        }

        float Rnd::rnd_flt(float min, float max)
        {
            std::uniform_real_distribution<float> dist(min, max);
            return dist(rg.at(omp_get_thread_num()));
        }

        float Rnd::rnd_dbl(float min, float max)
        {
            std::uniform_real_distribution<float> dist(min, max);
            return dist(rg.at(omp_get_thread_num()));
        }
        
        float Rnd::operator()(unsigned i) 
        {
            return rnd_dbl(0.0,i);
        }
        
        float Rnd::operator()() { return rnd_flt(0.0,1.0); }

        /*
	    template <class RandomAccessIterator>
	    void Rnd::shuffle (RandomAccessIterator first, RandomAccessIterator last)
	    {
	        for (auto i=(last-first)-1; i>0; --i) 
            {
	            std::uniform_int_distribution<decltype(i)> d(0,i);
		        swap (first.at(i), first[d(rg[omp_get_thread_num()])]);
	        }
	    }*/           
        
        /*template<typename Iter>                                    
        Iter Rnd::select_randomly(Iter start, Iter end) 
        {
            std::uniform_int_distribution<> dis(0, distance(start, end) - 1);
            advance(start, dis(rg[omp_get_thread_num()]));
            return start;
        }*/
       
        /*
        template<typename T>
        T Rnd::random_choice(const vector<T>& v)
        {
            //return a random element of a vector.          
            assert(v.size()>0 && " attemping to return random choice from empty vector");
            return *select_randomly(v.begin(),v.end());
        }*/

        /*
        template<typename T, typename D>
        T Rnd::random_choice(const vector<T>& v, const vector<D>& w )
        {
             //return a weighted random element of a vector
             
            if(w.size() == 0)
            {   
                cout<<"random_choice() w.size() = 0 Calling random_choice(v)\n";
                return random_choice(v);
            }
            if(w.size() != v.size())
            {   
                cout<<"WARN! random_choice() w.size() " << w.size() << "!= v.size() " 
                    << v.size() << ", Calling random_choice(v)\n";
                return random_choice(v);
            }
            else
            {
                assert(v.size() == w.size());
                std::discrete_distribution<size_t> dis(w.begin(), w.end());

                return v[dis(rg[omp_get_thread_num()])]; 
            }
        }*/
            
        float Rnd::gasdev()
        //Returns a normally distributed deviate with zero mean and unit variance
        {
            float ran = rnd_flt(-1,1);
            static int iset=0;
            static float gset;
            float fac,rsq,v1,v2;
            if (iset == 0) {// We don't have an extra deviate handy, so 
                do{
                    v1=float(2.0*rnd_flt(-1,1)-1.0); //pick two uniform numbers in the square ex
                    v2=float(2.0*rnd_flt(-1,1)-1.0); //tending from -1 to +1 in each direction,
                    rsq=v1*v1+v2*v2;	   //see if they are in the unit circle,
                } while (rsq >= 1.0 || rsq == 0.0); //and if they are not, try again.
                fac=float(sqrt(-2.0*log(rsq)/rsq));
            //Now make the Box-Muller transformation to get two normal deviates. Return one and
            //save the other for next time.
            gset=v1*fac;
            iset=1; //Set flag.
            return v2*fac;
            } 
            else 
            {		//We have an extra deviate handy,
                iset=0;			//so unset the flag,
                return gset;	//and return it.
            }
        }
        Rnd::~Rnd() {}
    }

}
