/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#include "n_split.h"

namespace FT{

	NodeSplit::NodeSplit()
	{
		name = "split";
		otype = 'b';
		arity['f'] = 1;
		arity['b'] = 0;
		complexity = 2;
        threshold = 0;
	}

    void NodeSplit::evaluate(const Data& data, Stacks& stack)
    {
        /* cout << "classification: " << data.classification << "\n"; */
        ArrayXd x1 = stack.f.pop();
        if (!data.validation && !data.y.size()==0)
            set_threshold(x1,data.y, data.classification);

        stack.b.push(x1 < threshold);
    }

    /// Evaluates the node symbolically
    void NodeSplit::eval_eqn(Stacks& stack)
    {
        stack.bs.push("(" + stack.fs.pop() + "<" + std::to_string(threshold) + ")");
    }
    
    NodeSplit* NodeSplit::clone_impl() const { return new NodeSplit(*this); };  
    NodeSplit* NodeSplit::rnd_clone_impl() const { return new NodeSplit(); };  
    
    void NodeSplit::set_threshold(ArrayXd& x, VectorXd& y, bool classification)
    {
        // for each unique value in x, calculate the reduction in the heuristic brought about by
        // splitting between that value and the next. 
        // set threshold according to the biggest reduction. 
        vector<double> s;
        for (unsigned i = 0; i < x.size(); ++i) s.push_back(x(i)); //(x.data(),x.size());
        vector<int> idx(s.size());
        std::iota(idx.begin(),idx.end(), 0);
        Map<ArrayXi> midx(idx.data(),idx.size());
        s = unique(s);
        double score = 0;
        double best_score = 0;
        /* cout << "s: " ; */ 
        /* for (auto ss : s) cout << ss << " " ; cout << "\n"; */
        for (unsigned i =0; i<s.size()-1; ++i)
        {

            double val = (s.at(i) + s.at(i+1)) / 2;
            /* cout << "split val: " << val << "\n"; */
            ArrayXi split_idx = (x < val).select(midx,-midx-1);
            // split data
            vector<double> d1, d2; 
            for (unsigned j=0; j< split_idx.size(); ++j)
            {
                if (split_idx(j) <0)
                    d2.push_back(y(-1-split_idx(j)));
                else
                    d1.push_back(y(split_idx(j)));
            }
            if (d1.empty() || d2.empty())
                continue;

            Map<VectorXd> map_d1(d1.data(), d1.size());  
            Map<VectorXd> map_d2(d2.data(), d2.size());  
            /* cout << "d1: " << map_d1.transpose() << "\n"; */
            /* cout << "d2: " << map_d2.transpose() << "\n"; */
            score = gain(map_d1, map_d2, classification);
            /* cout << "score: " << score << "\n"; */
            if (score < best_score || i == 0)
            {
                best_score = score;
                threshold = val;
            }
        }

        /* cout << "final threshold set to " << threshold << "\n"; */
    }
   
    double NodeSplit::gain(const VectorXd& lsplit, const VectorXd& rsplit, 
            bool classification)
    {
        double lscore, rscore, score;
        if (classification)
        {
            lscore = gini_impurity_index(lsplit);
            rscore = gini_impurity_index(rsplit);
            score = (lscore*double(lsplit.size()) + rscore*double(rsplit.size()))
                        /(double(lsplit.size()) + double(rsplit.size()));
        }
        else
        {
            lscore = variance(lsplit.array())/double(lsplit.size());
            rscore = variance(rsplit.array())/double(rsplit.size());
            score = lscore + rscore; 
        }

	    return score;
    }

    double NodeSplit::gini_impurity_index(const VectorXd& classes)
    {
        vector<double> uc = unique(classes);
        VectorXd class_weights(uc.size());
        for (auto c : uc){
            class_weights(c) = double((classes.cast<int>().array() == int(c)).count())
                                        /classes.size(); 
        }
        /* double total_weight=class_weights.sum(); */
        double gini = 1 - class_weights.dot(class_weights);

        return gini;
    }

    /* double NodeSplit::least_squares_deviation(const VectorXd& feats) */
    /* { */

    /*     return variance(feats.array()); */
    /*     /1* Map<VectorXd> map_weights(weights.vector, weights.size()); *1/ */
    /*     /1* Map<VectorXd> map_feats(feats.vector, weights.size()); *1/ */
    /*     /1* float64_t mean=map_weights.dot(map_feats); *1/ */
    /*     /1* total_weight=map_weights.sum(); *1/ */

    /*     /1* mean/=total_weight; *1/ */
    /*     /1* float64_t dev=0; *1/ */
    /*     /1* for (int32_t i=0;i<weights.vlen;i++) *1/ */
    /*     /1*     dev+=weights[i]*(feats[i]-mean)*(feats[i]-mean); *1/ */

    /*     /1* return dev/total_weight; *1/ */
    /* } */
}	
