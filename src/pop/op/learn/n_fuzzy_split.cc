/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_fuzzy_split.h"

namespace FT{

    namespace Pop{
        namespace Op{

	        template <>
	        NodeFuzzySplit<float>::NodeFuzzySplit()
	        {
	            name = "fuzzy_split";
	            arity['f'] = 1;
	            otype = 'b';
                complexity = 2;
                threshold = 0;
                train = false;	
	        }
	
	        template <>
	        NodeFuzzySplit<int>::NodeFuzzySplit()
	        {
	            name = "fuzzy_split_c";
	            arity['c'] = 1;
	            otype = 'b';
                complexity = 2;
                threshold = 0;
                train = false;	
	        }

             #ifndef USE_CUDA
            template <class T>
            void NodeFuzzySplit<T>::evaluate(const Data& data, State& state)
            {
                ArrayXf x1;
                        
                x1 = state.pop<T>().template cast<float>();
                    
                if (!data.validation && !data.y.size()==0 && train)
                    set_threshold(x1,data.y, data.classification);
                    
                if(arity['f'])
                    state.push<bool>(x1 < threshold);
                else
                    state.push<bool>(x1 == threshold);
            }
            #else
            template <class T>
            void NodeFuzzySplit<T>::evaluate(const Data& data, State& state)
            {
                ArrayXf x1(state.N);
                
                if(arity['f'])
                {
                    ArrayXf x(state.N);
                    state.copy_to_host(x.data(), (state.idx['f']-1)*state.N);
                    x1 = x.cast<float>();                    
                }
                else
                {
                    ArrayXi x(state.N);
                    state.copy_to_host(x.data(), (state.idx['c']-1)*state.N);
                    x1 = x.cast<float>();
                }
                    
                    
                if (!data.validation && !data.y.size()==0 && train)
                    set_threshold(x1,data.y, data.classification);
                    
                if(arity['f'])
                    GPU_FuzzySplit(state.dev_f, state.dev_b, state.idx['f'], 
                            state.idx[otype], state.N, threshold);
                else
                    GPU_FuzzySplit(state.dev_c, state.dev_b, state.idx['c'], 
                            state.idx[otype], state.N, threshold);
            }
            #endif

            /// Evaluates the node symbolically
            template <class T>
            void NodeFuzzySplit<T>::eval_eqn(State& state)
            {
                if(arity['f'])
                    state.push<bool>("(" + state.popStr<T>() + "<" + 
                            to_string(threshold, 4) + ")");
                else
                    state.push<bool>("(" + state.popStr<T>() + "==" + 
                            to_string(threshold, 4) + ")");
            }
            
            template <class T>
            NodeFuzzySplit<T>* NodeFuzzySplit<T>::clone_impl() const { 
                return new NodeFuzzySplit<T>(*this); };  
            
            template <class T>
            NodeFuzzySplit<T>* NodeFuzzySplit<T>::rnd_clone_impl() const { 
                return new NodeFuzzySplit<T>(); };  
            
            template <class T>
            void NodeFuzzySplit<T>::set_threshold(ArrayXf& x, VectorXf& y, 
                    bool classification)
            {
                /* cout << "setting threshold\n"; */
                // for each unique value in x, calculate the reduction in the 
                // heuristic brought about by
                // splitting between that value and the next. 
                // set threshold according to the biggest reduction. 
                vector<float> s;
                for (unsigned i = 0; i < x.size(); ++i) 
                    s.push_back(x(i)); //(x.data(),x.size());

                vector<float> unique_classes = unique(y);
                vector<int> idx(s.size());
                std::iota(idx.begin(),idx.end(), 0);
                Map<ArrayXi> midx(idx.data(),idx.size());
                s = unique(s);
                if (s.size() == 1)
                {
                    // if there is only one value, just set the threshold to 
                    // that
                    threshold = s.at(0);
                    return;
                }
                float score = 0;
                float best_score = 0;
                vector<float> neg_scores; // holds all scores for sampling
                vector<float> thresholds; // holds all scores for sampling
                /* cout << "s: " ; */ 
                /* for (auto ss : s) cout << ss << " " ; cout << "\n"; */
                /* cout << "x: " << x << "\n"; */
                /* cout << "y: " << y << "\n"; */
                /* cout << "threshold,score\n"; */
           
                for (unsigned i =0; i<s.size()-1; ++i)
                {

                    float val;
                    ArrayXi split_idx;
                    
                    if(arity['f'])
                    {
                        val = (s.at(i) + s.at(i+1)) / 2;
                        split_idx = (x < val).select(midx,-midx-1);
                    }
                    else
                    {
                        val = s.at(i);
                        split_idx = (x == val).select(midx,-midx-1);
                    }
                    // split data
                    vector<float> d1, d2; 
                    for (unsigned j=0; j< split_idx.size(); ++j)
                    {
                        if (split_idx(j) <0)
                            d2.push_back(y(-1-split_idx(j)));
                        else
                            d1.push_back(y(split_idx(j)));
                    }
                    if (d1.empty() || d2.empty())
                        continue;

                    Map<VectorXf> map_d1(d1.data(), d1.size());  
                    Map<VectorXf> map_d2(d2.data(), d2.size());  
                    /* cout << "d1: " << map_d1.transpose() << "\n"; */
                    /* cout << "d2: " << map_d2.transpose() << "\n"; */
                    score = gain(map_d1, map_d2, classification, 
                            unique_classes);

                    neg_scores.push_back(-score);
                    thresholds.push_back(val);
                    /* cout << "score: " << score << "\n"; */
                    /* cout << val << "," << score << "\n"; */
                    if (score < best_score || i == 0)
                    {
                        best_score = score;
                    }
                }
                if (thresholds.empty())
                {
                    /* cout << "threshold set to zero\n"; */
                    threshold = 0; 
                    return; 
                }
                else
                {
                    // choose a random threshold weighted by the scores
                    threshold  = r.random_choice(thresholds, neg_scores); 
                    int index = distance(thresholds.begin(), 
                            find(thresholds.begin(), thresholds.end(),
                                    threshold));
                    /* cout << "index: " << index << "\n"; */
                    /* cout << "final threshold set to " << threshold */ 
                    /*     << " with score " << -neg_scores.at(index)<< "\n"; */ 
                }
            }
           
            template <class T>
            float NodeFuzzySplit<T>::gain(const VectorXf& lsplit, 
                    const VectorXf& rsplit, 
                    bool classification, vector<float> unique_classes)
            {
                float lscore, rscore, score;
                if (classification)
                {
                    lscore = gini_impurity_index(lsplit, unique_classes);
                    rscore = gini_impurity_index(rsplit, unique_classes);
                    /* cout << "lscore: " << lscore << "\n"; */
                    /* cout << "rscore: " << rscore << "\n"; */
                    score = (lscore*float(lsplit.size()) + 
                            rscore*float(rsplit.size()))
                                /(float(lsplit.size()) + float(rsplit.size()));
                }
                else
                {
                    lscore = variance(lsplit.array())/float(lsplit.size());
                    rscore = variance(rsplit.array())/float(rsplit.size());
                    score = lscore + rscore; 
                }

	            return score;
            }

            template <class T>
            float NodeFuzzySplit<T>::gini_impurity_index(
                    const VectorXf& classes, vector<float> uc)
            {
                VectorXf class_weights(uc.size());
                for (auto c : uc){
                    class_weights(static_cast<Eigen::Index>(c)) = 0;
                    class_weights(static_cast<Eigen::Index>(c)) = float(
                            (classes.cast<int>().array() == int(c)).count()
                            )/classes.size();
                    /* cout << "class_weights for " << c << ": " 
                     * << class_weights(c) << "\n"; */
                }
                /* float total_weight=class_weights.sum(); */
                float gini = 1 - class_weights.dot(class_weights);

                return gini;
            }
        }
    }
}	
